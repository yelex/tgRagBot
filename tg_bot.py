import json
import os
import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ContextTypes,
)

from flower_logic import FlowerLogic, create_price_ranges, format_bouquet_message, get_bouquet_image

# Настройка логгера
_log_dir = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, "flower_bot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPERATOR_CHAT_ID = os.getenv('OPERATOR_CHAT_ID')
FEEDBACK_DELAY_HOURS = float(os.getenv('FEEDBACK_DELAY_HOURS', '2'))


class TelegramBot:
    def __init__(self):
        self.flower_logic = FlowerLogic()
        self.application = None
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        self.scheduler = AsyncIOScheduler()
        self._feedback_scheduled: Set[int] = set()  # user_id → уже запланирован фидбэк
        logger.info("Бот инициализирован")

    async def setup_application(self):
        """Асинхронная инициализация и настройка приложения бота"""
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        await self.application.initialize()
        self.register_handlers()
        self.scheduler.start()
        logger.info("Приложение бота настроено и инициализировано")

    def register_handlers(self):
        """Регистрация обработчиков команд"""
        if not self.application:
            logger.error("Application не инициализировано")
            return
            
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prices", self.show_price_ranges))
        self.application.add_handler(CallbackQueryHandler(self.handle_price_range, pattern="^price_"))
        self.application.add_handler(CallbackQueryHandler(self.handle_more_bouquets, pattern="^more_"))
        self.application.add_handler(CallbackQueryHandler(self.handle_feedback_callback, pattern="^feedback_"))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        logger.info("Обработчики команд зарегистрированы")

    def _get_conversation_history(self, user_id: int) -> List[Dict[str, str]]:
        return self.conversation_history.get(user_id, [])

    def _add_to_history(self, user_id: int, role: str, message: str):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        if len(self.conversation_history[user_id]) >= 10:
            self.conversation_history[user_id] = self.conversation_history[user_id][-9:]
        self.conversation_history[user_id].append({"role": role, "content": message})

    # ── Background jobs ───────────────────────────────────────────────────────

    def _schedule_feedback(self, user_id: int, order_id: Optional[int] = None):
        """Планирует запрос фидбэка через FEEDBACK_DELAY_HOURS после доставки."""
        if user_id in self._feedback_scheduled:
            return
        self._feedback_scheduled.add(user_id)
        run_at = datetime.now() + timedelta(hours=FEEDBACK_DELAY_HOURS)
        job_id = f"feedback_{user_id}"
        self.scheduler.add_job(
            self._send_feedback_request,
            trigger="date",
            run_date=run_at,
            args=[user_id, order_id],
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "Фидбэк запланирован: user_id=%s order_id=%s at=%s",
            user_id, order_id, run_at.strftime("%H:%M"),
        )

    async def _send_feedback_request(self, user_id: int, order_id: Optional[int]):
        """Фоновая задача: отправляет запрос оценки после доставки."""
        if not self.application:
            return
        self._feedback_scheduled.discard(user_id)
        order_line = f" (заказ #{order_id})" if order_id else ""
        text = (
            f"Ваш букет{order_line} уже у получателя!\n\n"
            "Всё прошло хорошо?"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Да, всё отлично!", callback_data=f"feedback_ok_{user_id}"),
                InlineKeyboardButton("Есть вопрос", callback_data=f"feedback_issue_{user_id}"),
            ]
        ])
        try:
            await self.application.bot.send_message(
                chat_id=user_id,
                text=text,
                reply_markup=keyboard,
            )
            logger.info("Фидбэк-запрос отправлен: user_id=%s", user_id)
        except Exception as e:
            logger.error("Не удалось отправить фидбэк: user_id=%s error=%s", user_id, e)

    async def handle_feedback_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data  # feedback_ok_<uid> или feedback_issue_<uid>

        if data.startswith("feedback_ok_"):
            await query.edit_message_text(
                "Рады слышать! Если захотите повторить или порекомендовать нас — будем благодарны.\n\n"
                "Хорошего дня!"
            )
            logger.info("Фидбэк положительный: %s", data)
        elif data.startswith("feedback_issue_"):
            user_id = update.effective_user.id
            user_name = update.effective_user.username or update.effective_user.full_name or ""
            await query.edit_message_text(
                "Понял, сейчас разберёмся. Напишите, пожалуйста, что случилось."
            )
            # Уведомляем оператора
            await self._notify_operator(
                context=context,
                user_id=user_id,
                user_name=user_name,
                escalation={"reason": "feedback_negative"},
                history=self._get_conversation_history(user_id),
            )
            logger.info("Негативный фидбэк → оператор: user_id=%s", user_id)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.conversation_history[user_id] = []

        welcome_text = """🌸 *Добро пожаловать в цветочный магазин!* 🌸

Я помогу вам подобрать идеальный букет. Вы можете:
- Написать ваш запрос (например, _Ищу розы до 10000 рублей_)
- Выбрать ценовой диапазон
- Попросить рекомендацию"""

        self._add_to_history(user_id, "assistant", welcome_text)
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
        await self.show_price_ranges(update, context)

    async def show_price_ranges(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        price_ranges = create_price_ranges()
        keyboard = [[InlineKeyboardButton(text, callback_data=f"price_{data}")] for text, data in price_ranges]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="*Выберите ваш бюджет:*",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        logger.info(f"Пользователю {update.effective_user.id} показаны ценовые диапазоны")

    async def handle_price_range(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        selected_range = query.data

        logger.info(f"Пользователь {user_id} выбрал диапазон: {selected_range}")

        # Извлекаем максимальную цену из callback_data
        price_part = selected_range.split('_')[1]
        if '+' in price_part:
            max_price = float('inf')
        else:
            max_price = float(price_part)

        bouquets = self.flower_logic.filter_bouquets_by_price(max_price)

        if not bouquets:
            await query.edit_message_text("К сожалению, в этом ценовом диапазоне букетов нет 😢")
            logger.warning(f"Для пользователя {user_id} не найдено букетов в диапазоне {max_price}")
            return

        for bouquet in bouquets[:3]:
            message = format_bouquet_message(bouquet)
            image_url = get_bouquet_image(bouquet)
            if image_url:
                try:
                    await query.message.reply_photo(
                        photo=image_url,
                        caption=message,
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.warning(f"Не удалось отправить фото {image_url}: {e}")
                    await query.message.reply_text(
                        text=message,
                        parse_mode="Markdown",
                        disable_web_page_preview=True
                    )
            else:
                await query.message.reply_text(
                    text=message,
                    parse_mode="Markdown",
                    disable_web_page_preview=True
                )

        if len(bouquets) > 3:
            await query.message.reply_text(
                text=f"Показано 3 из {len(bouquets)} вариантов. Уточните запрос для более точного подбора.",
                parse_mode="Markdown"
            )

    async def handle_more_bouquets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id

        # callback_data = "more_{start}_{remaining}"
        parts = query.data.split("_")
        start = int(parts[1])
        remaining = int(parts[2])

        more_text = self.flower_logic.get_more_bouquets(user_id, start, remaining)
        if more_text:
            await query.message.reply_text(more_text, parse_mode="Markdown")
        else:
            await query.message.reply_text("Больше вариантов нет.", parse_mode="Markdown")

    def _find_first_bouquet(self, text: str):
        """Ищет первый букет, упомянутый в тексте."""
        for b in self.flower_logic.bouquets_data:
            if b["Название"].lower() in text.lower():
                return b
        return None

    @staticmethod
    def _clean_response(text: str):
        """Очищает ответ от служебных мета-маркеров. Возвращает (cleaned_text, keyboard_or_None, escalation_or_None)."""
        keyboard = None
        escalation = None

        kb_match = re.search(r'\|\|keyboard:(\[.*?\])\|\|', text)
        if kb_match:
            try:
                keyboard = json.loads(kb_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        esc_match = re.search(r'\|\|escalate:(\{.*?\})\|\|', text)
        if esc_match:
            try:
                escalation = json.loads(esc_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        cleaned = re.sub(r'\|\|phase:[a-zA-Z0-9_]+\|\|', '', text)
        cleaned = re.sub(r'\|\|data:\{.*?\}\|\|', '', cleaned)
        cleaned = re.sub(r'\|\|more:\d+:\d+\|\|', '', cleaned)
        cleaned = re.sub(r'\|\|keyboard:\[.*?\]\|\|', '', cleaned)
        cleaned = re.sub(r'\|\|escalate:\{.*?\}\|\|', '', cleaned)
        return cleaned.strip(), keyboard, escalation

    async def _notify_operator(
        self,
        context,
        user_id: int,
        user_name: str,
        escalation: dict,
        history: List[Dict[str, str]],
    ):
        if not OPERATOR_CHAT_ID:
            logger.warning("OPERATOR_CHAT_ID не задан, уведомление оператору пропущено")
            return

        reason_labels = {
            "complaint": "Жалоба клиента",
            "photo_dispute": "Спор по фото",
            "delivery_conflict": "Конфликт доставки",
            "other": "Нестандартная ситуация",
        }
        reason = reason_labels.get(escalation.get("reason", "other"), escalation.get("reason", "—"))

        order_line = ""
        if escalation.get("order_id"):
            parts = [f"#{escalation['order_id']}"]
            if escalation.get("bouquet_name"):
                parts.append(escalation["bouquet_name"])
            if escalation.get("budget_max"):
                parts.append(f"до {int(escalation['budget_max'])} ₽")
            order_line = f"\n💐 Заказ: {' — '.join(parts)}"

        address_line = ""
        if escalation.get("address"):
            address_line = f"\n📍 Адрес: {escalation['address']}"

        last_msgs = []
        for msg in history[-5:]:
            role_label = "Клиент" if msg.get("role") == "user" else "Бот"
            content = re.sub(r'\|\|.*?\|\|', '', msg.get("content", "")).strip()
            if content:
                last_msgs.append(f"{role_label}: {content[:120]}")
        history_block = "\n".join(last_msgs) if last_msgs else "—"

        text = (
            f"🚨 Нужна помощь оператора\n\n"
            f"👤 Клиент: @{user_name} (ID: {user_id}){order_line}{address_line}\n"
            f"📋 Причина: {reason}\n\n"
            f"Последние сообщения:\n{history_block}\n\n"
            f"🔗 Написать клиенту: tg://user?id={user_id}"
        )

        try:
            await context.bot.send_message(
                chat_id=OPERATOR_CHAT_ID,
                text=text,
            )
            logger.info("Уведомление оператору отправлено: user_id=%s reason=%s", user_id, escalation.get("reason"))
        except Exception as e:
            logger.error("Не удалось отправить уведомление оператору: %s", e)

    @staticmethod
    def _sanitize_for_markdown(text: str) -> str:
        """Экранирует символы, ломающие Markdown Telegram.
        
        Telegram Markdown чувствителен к: _ * ` [ ] ~ >
        Экранируем только те, что НЕ являются частью Markdown-разметки.
        """
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            # Пропускаем уже корректные Markdown-конструкции
            if ch == '*' and i + 1 < len(text) and text[i + 1] == '*':
                # **bold** — корректная конструкция
                result.append('**')
                i += 2
                continue
            if ch == '_':
                # В URL _ это часть ссылки, не Markdown
                # Просто экранируем все _, они в Markdown — курсив, 
                # но если_не_закрыты — ломают парсинг
                result.append('\\_')
                i += 1
                continue
            # Экранируем проблемные символы вне конструкций
            if ch in '`[]~>':
                result.append('\\' + ch)
                i += 1
                continue
            result.append(ch)
            i += 1
        return ''.join(result)

    async def _send_response(self, update: Update, text: str, reply_markup=None):
        """Отправляет ответ: если возможно — с фото, иначе просто текст.
        
        Telegram ограничения:
        - caption фото: 1024 символа
        - сообщение: 4096 символов
        
        Важно: перед отправкой очищаем мета-маркеры.
        """
        # Очищаем от служебных маркеров
        clean_text, kb_list, _ = self._clean_response(text)
        # Если reply_markup ещё не передан, а kb_list есть — создаём ReplyKeyboardMarkup
        if reply_markup is None and kb_list:
            reply_markup = ReplyKeyboardMarkup(
                [[KeyboardButton(opt)] for opt in kb_list],
                resize_keyboard=True,
                one_time_keyboard=True,
            )

        # Для URL внутри текста: ищем букет
        first_bouquet = self._find_first_bouquet(clean_text)
        image_url = get_bouquet_image(first_bouquet) if first_bouquet else None

        # Если текст короткий и есть фото — отправляем как caption
        if image_url and len(clean_text) <= 1024:
            try:
                await update.message.reply_photo(
                    photo=image_url,
                    caption=clean_text,
                    reply_markup=reply_markup,
                )
                return
            except Exception as e:
                logger.warning(f"Не удалось отправить фото с caption: {e}")

        # Если фото есть, но текст длинный — отправляем фото и текст отдельно
        if image_url:
            try:
                await update.message.reply_photo(
                    photo=image_url,
                )
            except Exception as e:
                logger.warning(f"Не удалось отправить фото: {e}")

        # Отправляем текст (с разбивкой, если слишком длинный)
        if len(clean_text) <= 4096:
            await update.message.reply_text(
                clean_text,
                reply_markup=reply_markup,
            )
        else:
            # Разбиваем на части по ~4000 символов
            max_len = 4000
            parts = []
            current = ""
            for line in clean_text.split('\n'):
                if len(current) + len(line) + 1 > max_len:
                    parts.append(current)
                    current = line
                else:
                    if current:
                        current += '\n' + line
                    else:
                        current = line
            if current:
                parts.append(current)

            for i, part in enumerate(parts):
                rm = reply_markup if i == 0 else None
                await update.message.reply_text(
                    part,
                    reply_markup=rm,
                )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_name = (
            update.effective_user.username
            or update.effective_user.full_name
            or update.effective_user.first_name
        )
        user_input = update.message.text
        self._add_to_history(user_id, "user", user_input)
        logger.info(f"Получено сообщение от {user_id} ({user_name}): {user_input}")

        try:
            raw_history = self._get_conversation_history(user_id)
            response = self.flower_logic.get_bouquet_recommendation(
                user_input=user_input,
                user_id=user_id,
                user_name=user_name,
                conversation_history=raw_history
            )

            self._add_to_history(user_id, "assistant", response)

            # Проверяем маркер эскалации — уведомляем оператора до отправки клиенту
            esc_match = re.search(r'\|\|escalate:(\{.*?\})\|\|', response)
            if esc_match:
                try:
                    escalation = json.loads(esc_match.group(1))
                    await self._notify_operator(
                        context=context,
                        user_id=user_id,
                        user_name=user_name,
                        escalation=escalation,
                        history=self._get_conversation_history(user_id),
                    )
                except Exception as e:
                    logger.error("Ошибка при обработке эскалации: %s", e)

            # Планируем фидбэк если заказ доставлен
            if "||phase:N15_close||" in response:
                data_match = re.search(r'\|\|data:(\{.*?\})\|\|', response)
                order_id = None
                if data_match:
                    try:
                        order_id = json.loads(data_match.group(1)).get("order_id")
                    except (json.JSONDecodeError, ValueError):
                        pass
                self._schedule_feedback(user_id, order_id)

            # Проверяем, есть ли маркер "показать ещё"
            more_match = re.search(r'\|\|more:(\d+):(\d+)\|\|', response)
            reply_markup = None
            text_to_send = response

            if more_match:
                remaining = int(more_match.group(1))
                start = int(more_match.group(2))
                text_to_send = response.replace(more_match.group(0), "").strip()
                keyboard = [[InlineKeyboardButton(
                    f"Показать ещё {remaining}",
                    callback_data=f"more_{start}_{remaining}"
                )]]
                reply_markup = InlineKeyboardMarkup(keyboard)

            await self._send_response(update, text_to_send, reply_markup)
            logger.info(f"Отправлен ответ пользователю {user_id}")

        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
            await update.message.reply_text(
                "Произошла ошибка при обработке запроса. Попробуйте позже."
            )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_name = (
            update.effective_user.username
            or update.effective_user.full_name
            or update.effective_user.first_name
        )
        caption = update.message.caption or ""
        logger.info(f"Получено фото от {user_id} ({user_name}), caption={caption!r}")

        try:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = bytes(await file.download_as_bytearray())

            raw_history = self._get_conversation_history(user_id)
            response = self.flower_logic.handle_photo_message(
                user_id=user_id,
                user_name=user_name,
                photo_bytes=photo_bytes,
                caption=caption,
                conversation_history=raw_history,
            )
            self._add_to_history(user_id, "user", f"[фото] {caption}".strip())
            self._add_to_history(user_id, "assistant", response)
            await self._send_response(update, response)
            logger.info(f"Ответ на фото отправлен пользователю {user_id}")

        except Exception as e:
            logger.error(f"Ошибка при обработке фото: {e}", exc_info=True)
            await update.message.reply_text(
                "Не смог обработать фото. Попробуйте описать букет словами."
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """📌 *Как пользоваться ботом:*

1. Напишите что вы ищете (например: _розы до 10000 рублей_ или _пионы для девушки_)
2. Или выберите ценовой диапазон
3. Бот предложит вам подходящие варианты

*Доступные команды:*
/start - начать диалог
/help - показать эту справку
/prices - показать ценовые диапазоны"""
        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Пользователь {update.effective_user.id} запросил помощь")

    def run(self):
        """Запуск бота"""
        logger.info("Запуск бота...")
        
        # Создаем и запускаем event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Асинхронно настраиваем приложение
            loop.run_until_complete(self.setup_application())
            
            if not self.application:
                logger.critical("Не удалось настроить приложение бота")
                return
                
            # Запускаем polling
            self.application.run_polling()
        except Exception as e:
            logger.critical(f"Критическая ошибка при запуске бота: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()


if __name__ == '__main__':
    bot = TelegramBot()
    bot.run()