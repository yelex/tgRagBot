import json
import os
import logging
import asyncio
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/logs/flower_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')


class TelegramBot:
    def __init__(self):
        self.flower_logic = FlowerLogic()
        self.application = None
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        logger.info("Бот инициализирован")

    async def setup_application(self):
        """Асинхронная инициализация и настройка приложения бота"""
        # Создаем Application
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Асинхронная инициализация бота
        await self.application.initialize()
        
        # Регистрируем обработчики
        self.register_handlers()
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
        """Очищает ответ от служебных мета-маркеров. Возвращает (cleaned_text, keyboard_or_None)."""
        keyboard = None
        kb_match = re.search(r'\|\|keyboard:(\[.*?\])\|\|', text)
        if kb_match:
            try:
                keyboard = json.loads(kb_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        cleaned = re.sub(r'\|\|phase:[a-zA-Z0-9_]+\|\|', '', text)
        cleaned = re.sub(r'\|\|data:\{.*?\}\|\|', '', cleaned)
        cleaned = re.sub(r'\|\|more:\d+:\d+\|\|', '', cleaned)
        cleaned = re.sub(r'\|\|keyboard:\[.*?\]\|\|', '', cleaned)
        return cleaned.strip(), keyboard

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
        # Очищаем от служебных маркеров (теперь функция возвращает кортеж)
        clean_text, kb_list = self._clean_response(text)
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