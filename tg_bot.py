import os
import logging
from typing import Dict, List
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

from flower_logic import FlowerLogic, create_price_ranges, format_bouquet_message
from langchain.schema import HumanMessage, AIMessage, BaseMessage

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("flower_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')


class TelegramBot:
    def __init__(self):
        self.flower_logic = FlowerLogic()
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}

        self.register_handlers()
        logger.info("Бот инициализирован")

    def register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prices", self.show_price_ranges))
        self.application.add_handler(CallbackQueryHandler(self.handle_price_range, pattern="^price_"))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        logger.info("Обработчики команд зарегистрированы")

    def _get_conversation_history(self, chat_id: int) -> List[Dict[str, str]]:
        return self.conversation_history.get(chat_id, [])

    def _add_to_history(self, chat_id: int, role: str, message: str):
        if chat_id not in self.conversation_history:
            self.conversation_history[chat_id] = []
        if len(self.conversation_history[chat_id]) >= 10:
            self.conversation_history[chat_id] = self.conversation_history[chat_id][-9:]
        self.conversation_history[chat_id].append({"role": role, "content": message})

    def _convert_history_to_messages(self, history: List[Dict[str, str]]) -> List[BaseMessage]:
        messages = []

        if not isinstance(history, list):
            logger.error(f"_convert_history_to_messages: history не список: {type(history)}")
            return messages

        for entry in history:
            if not isinstance(entry, dict):
                logger.warning(f"_convert_history_to_messages: элемент не словарь: {entry}")
                continue

            role = entry.get("role")
            content = entry.get("content")

            if not isinstance(content, str):
                logger.warning(f"_convert_history_to_messages: content не строка: {content}")
                continue

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    def _escape_markdown(self, text: str) -> str:
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return ''.join(f'\\{char}' if char in escape_chars else char for char in text)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self.conversation_history[chat_id] = []

        welcome_text = """
🌸 *Добро пожаловать в цветочный магазин!* 🌸

Я помогу вам подобрать идеальный букет. Вы можете:
- Написать ваш запрос (например, _Ищу розы до 10000 рублей_)
- Выбрать ценовой диапазон
- Попросить рекомендацию
"""

        self._add_to_history(chat_id, "assistant", welcome_text)
        await update.message.reply_text(self._escape_markdown(welcome_text), parse_mode="MarkdownV2")
        await self.show_price_ranges(update, context)

    async def show_price_ranges(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        price_ranges = create_price_ranges()
        keyboard = [[InlineKeyboardButton(text, callback_data=f"price_{data}")] for text, data in price_ranges]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=r"*Выберите ваш бюджет:*",
            reply_markup=reply_markup,
            parse_mode="MarkdownV2"
        )
        logger.info(f"Пользователю {update.effective_user.id} показаны ценовые диапазоны")

    async def handle_price_range(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = update.effective_user.id
        selected_range = query.data

        logger.info(f"Пользователь {user_id} выбрал диапазон: {selected_range}")

        max_price = float(selected_range.split('_')[1].replace('+', ''))
        bouquets = self.flower_logic.filter_bouquets_by_price(max_price)

        if not bouquets:
            await query.edit_message_text("К сожалению, в этом ценовом диапазоне букетов нет 😢")
            logger.warning(f"Для пользователя {user_id} не найдено букетов в диапазоне {max_price}")
            return

        for bouquet in bouquets[:3]:
            message = self._escape_markdown(format_bouquet_message(bouquet))
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=message,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True
            )

        if len(bouquets) > 3:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=r"Показано 3 из {} вариантов\. Уточните запрос для более точного подбора\.".format(len(bouquets)),
                parse_mode="MarkdownV2"
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        user_input = update.message.text
        self._add_to_history(chat_id, "user", user_input)
        logger.info(f"Получено сообщение от {chat_id}: {user_input}")

        try:
            raw_history = self._get_conversation_history(chat_id)
            converted_history = self._convert_history_to_messages(raw_history)

            logger.debug(f"История перед отправкой в GigaChat: {[type(m) for m in converted_history]}")

            response = self.flower_logic.get_bouquet_recommendation(
                user_input=user_input,
                conversation_history=converted_history
            )

            self._add_to_history(chat_id, "assistant", response)

            await update.message.reply_text(
                self._escape_markdown(response),
                parse_mode="MarkdownV2"
            )
            logger.info(f"Отправлен ответ пользователю {chat_id}")

        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            await update.message.reply_text(
                self._escape_markdown("Произошла ошибка при обработке запроса. Попробуйте позже."),
                parse_mode="MarkdownV2"
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = self._escape_markdown(r"""
📌 *Как пользоваться ботом:*

1\. Напишите что вы ищете \(например: _розы до 10000 рублей_ или _пионы для девушки_\)
2\. Или выберите ценовой диапазон
3\. Бот предложит вам подходящие варианты

*Доступные команды:*
/start \- начать диалог
/help \- показать эту справку
/prices \- показать ценовые диапазоны
""")
        await update.message.reply_text(help_text, parse_mode="MarkdownV2")
        logger.info(f"Пользователь {update.effective_user.id} запросил помощь")

    def run(self):
        logger.info("Запуск бота...")
        self.application.run_polling()


if __name__ == '__main__':
    try:
        bot = TelegramBot()
        logger.info("Бот успешно запущен")
        bot.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске бота: {e}")
