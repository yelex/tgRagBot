import os
import logging
import asyncio
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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
            if more_match:
                remaining = int(more_match.group(1))
                start = int(more_match.group(2))
                # Убираем маркер из текста
                clean_response = response.replace(more_match.group(0), "").strip()
                keyboard = [[InlineKeyboardButton(
                    f"Показать ещё {remaining}",
                    callback_data=f"more_{start}_{remaining}"
                )]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                # Пробуем найти первый букет из ответа и отправить его фото
                first_bouquet = None
                for b in self.flower_logic.bouquets_data:
                    if b["Название"].lower() in clean_response.lower():
                        first_bouquet = b
                        break
                if first_bouquet:
                    image_url = get_bouquet_image(first_bouquet)
                    if image_url:
                        try:
                            await update.message.reply_photo(
                                photo=image_url,
                                caption=clean_response,
                                parse_mode="Markdown",
                                reply_markup=reply_markup
                            )
                        except Exception as e:
                            logger.warning(f"Не удалось отправить фото {image_url}: {e}")
                            await update.message.reply_text(
                                clean_response,
                                parse_mode="Markdown",
                                reply_markup=reply_markup
                            )
                    else:
                        await update.message.reply_text(
                            clean_response,
                            parse_mode="Markdown",
                            reply_markup=reply_markup
                        )
                else:
                    await update.message.reply_text(
                        clean_response,
                        parse_mode="Markdown",
                        reply_markup=reply_markup
                    )
            else:
                # Пробуем найти первый букет из ответа и отправить его фото
                first_bouquet = None
                for b in self.flower_logic.bouquets_data:
                    if b["Название"].lower() in response.lower():
                        first_bouquet = b
                        break
                if first_bouquet:
                    image_url = get_bouquet_image(first_bouquet)
                    if image_url:
                        try:
                            await update.message.reply_photo(
                                photo=image_url,
                                caption=response,
                                parse_mode="Markdown"
                            )
                        except Exception as e:
                            logger.warning(f"Не удалось отправить фото {image_url}: {e}")
                            await update.message.reply_text(
                                response,
                                parse_mode="Markdown"
                            )
                    else:
                        await update.message.reply_text(
                            response,
                            parse_mode="Markdown"
                        )
                else:
                    await update.message.reply_text(
                        response,
                        parse_mode="Markdown"
                    )
            logger.info(f"Отправлен ответ пользователю {user_id}")

        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            await update.message.reply_text(
                "Произошла ошибка при обработке запроса. Попробуйте позже.",
                parse_mode="Markdown"
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