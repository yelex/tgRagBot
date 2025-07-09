import os
import logging
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

from flower_logic import FlowerLogic

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flower_bot.log"),  # Запись в файл
        logging.StreamHandler()  # Вывод в консоль
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')


class TelegramBot:
    def __init__(self):
        logger.info("Инициализация бота...")
        self.flower_logic = FlowerLogic()
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        self.register_handlers()
        logger.info("Бот успешно инициализирован")

    def register_handlers(self):
        """Регистрирует обработчики команд"""
        logger.info("Регистрация обработчиков команд...")
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prices", self.show_price_ranges))
        self.application.add_handler(CallbackQueryHandler(self.handle_price_range, pattern="^price_"))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        logger.info("Обработчики команд зарегистрированы")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        logger.info(f"Пользователь {user.id} ({user.username or 'без username'}) запустил бота")

        welcome_text = (
            "🌸 Добро пожаловать в цветочный магазин! 🌸\n\n"
            "Я помогу вам подобрать идеальный букет. Вы можете:\n"
            "- Написать ваш запрос (например, 'Ищу розы до 10000 рублей')\n"
            "- Выбрать ценовой диапазон\n"
            "- Попросить рекомендацию"
        )

        await update.message.reply_text(welcome_text)
        await self.show_price_ranges(update, context)
        logger.info("Приветственное сообщение отправлено")

    async def show_price_ranges(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показывает кнопки с ценовыми диапазонами"""
        logger.info("Показ ценовых диапазонов...")
        price_ranges = self.flower_logic.create_price_ranges()
        keyboard = [
            [InlineKeyboardButton(text, callback_data=f"price_{data}")]
            for text, data in price_ranges
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выберите ваш бюджет:",
            reply_markup=reply_markup
        )
        logger.info("Кнопки с ценовыми диапазонами отправлены")

    async def handle_price_range(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает выбор ценового диапазона"""
        query = update.callback_query
        await query.answer()

        selected_range = query.data
        logger.info(f"Выбран ценовой диапазон: {selected_range}")

        max_price = float(selected_range.split('_')[1].replace('+', ''))
        bouquets = self.flower_logic.filter_bouquets_by_price(max_price)

        if not bouquets:
            logger.warning(f"Букеты не найдены для диапазона: {max_price}")
            await query.edit_message_text("К сожалению, в этом ценовом диапазоне букетов нет 😢")
            return

        logger.info(f"Найдено {len(bouquets)} букетов в диапазоне до {max_price} руб.")

        # Отправляем первые 3 букета
        for bouquet in bouquets[:3]:
            message = self.flower_logic.format_bouquet_message(bouquet)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message,
                parse_mode='Markdown'
            )

        if len(bouquets) > 3:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Показано 3 из {len(bouquets)} вариантов. Уточните запрос для более точного подбора."
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает текстовые сообщения пользователя"""
        user_input = update.message.text
        user = update.effective_user
        logger.info(f"Получено сообщение от {user.id} ({user.username or 'без username'}): {user_input}")

        try:
            response = self.flower_logic.get_bouquet_recommendation(user_input, user_id=user.id)
            await update.message.reply_text(response)
            logger.info("Ответ успешно отправлен пользователю")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {str(e)}")
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        user = update.effective_user
        logger.info(f"Пользователь {user.id} запросил помощь")

        help_text = (
            "📌 Как пользоваться ботом:\n\n"
            "1. Напишите что вы ищете (например: 'розы до 10000 рублей' или 'пионы для девушки')\n"
            "2. Или выберите ценовой диапазон\n"
            "3. Бот предложит вам подходящие варианты\n\n"
            "Доступные команды:\n"
            "/start - начать диалог\n"
            "/help - показать эту справку\n"
            "/prices - показать ценовые диапазоны"
        )

        await update.message.reply_text(help_text)
        logger.info("Справка отправлена пользователю")

    def run(self):
        """Запускает бота"""
        logger.info("Запуск бота...")
        try:
            self.application.run_polling()
        except Exception as e:
            logger.critical(f"Критическая ошибка при работе бота: {str(e)}")
            raise


if __name__ == '__main__':
    try:
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Не удалось запустить бота: {str(e)}")