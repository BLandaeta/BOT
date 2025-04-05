import os
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process
from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext

# Cargar variables de entorno
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

LOGO_URL = os.getenv("LOGO_URL")

MEMORY_FILE = "memoria.json"
CUSTOM_RESPONSES_FILE = "preguntas.json"
MEMORY_LIMIT = 1000  # Número máximo de interacciones recordadas

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=GEMINI_API_KEY)

# Cargar respuestas personalizadas
def load_custom_responses():
    if not os.path.exists(CUSTOM_RESPONSES_FILE):
        return []
    with open(CUSTOM_RESPONSES_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("preguntas", [])

def find_custom_response(user_text):
    responses = load_custom_responses()
    user_embedding = model.encode(user_text, convert_to_tensor=True)
    
    best_response = None
    best_score = 0.0
    
    for entry in responses:
        for q in entry["keys"]:
            q_embedding = model.encode(q, convert_to_tensor=True)
            score = util.pytorch_cos_sim(user_embedding, q_embedding).item()
            if score > best_score:
                best_score = score
                best_response = entry["response"]
    
    if best_score >= 0.85:  # Umbral de similitud
        return best_response
    
    return None

# Funciones de historial de conversación
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4, ensure_ascii=False)

def load_base_conversation():
    """Carga la conversación base desde un archivo JSON."""
    if not os.path.exists('conversacion_base.json'):
        return []  # Si el archivo no existe, devuelve una lista vacía
    
    with open('conversacion_base.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("conversacion_base", [])

def initialize_user_memory(user_id):
    """Inicializa la memoria de un nuevo usuario con la conversación base cargada desde el archivo."""
    base_conversation = load_base_conversation()
    return base_conversation

# Obtener respuesta de Gemini con historial de usuario
def get_gemini_response(user_id, message):
    memory = load_memory()

    if str(user_id) not in memory:
        # Si el usuario es nuevo, inicializamos su memoria con una conversación base
        memory[str(user_id)] = initialize_user_memory(user_id)
        save_memory(memory)  # Guardamos la memoria actualizada

    user_memory = memory[str(user_id)]

    if len(user_memory) > MEMORY_LIMIT:
        user_memory = user_memory[-MEMORY_LIMIT:]  # Limitamos la memoria si se excede el límite

    chat_history = "\n".join(user_memory)
    prompt = f"Historial de conversación:\n{chat_history}\nUsuario: {message}\nGemini:"

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    reply = response.text if response.text else "No entendí la pregunta."

    user_memory.append(f"Usuario: {message}")
    user_memory.append(f"Gemini: {reply}")
    memory[str(user_id)] = user_memory
    save_memory(memory)

    return reply


# Manejo de mensajes y respuestas personalizadas
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    user_text = update.message.text.strip()

    response = find_custom_response(user_text)

    if not response:
        response = get_gemini_response(user_id, user_text)

    await update.message.reply_text(response)


# Comando Start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "¡Hola! Soy TransLanda bot 🤖 utiliza /comandos para ver todas mis funciones.")


# Comando Help
async def comandos (update: Update, context: CallbackContext):
    """Muestra los comandos del BOT"""
    help_text = (
        "*Comandos Disponibles:*\n\n"
        "** Inicia el BOT 🤖**  ```/start```\n"
        "** Comandos disponibles ⚙️**  ```/comandos```\n"
        "** Contacta con Nosotros  📝**  ```/contacto```\n"
        "** Ver los viajes disponibles  🚍**  ```/viajes```\n"
        "** Muestra la tasa del Dólar 💵**  ```/dolar```\n"
        "** Muestra el logo de *TransLanda* 📷**  ```/logo```\n"
        "** Busca imagen en Google 🔍**  ```/img+prompt```\n"
        "\n*Escribe un mensaje para preguntar lo que quieras a Gemini 1\\.5 Pro\\.*"
    )
    await update.message.reply_text(help_text, parse_mode="MarkdownV2")


# Comando Help
async def contacto (update: Update, context: CallbackContext):
    """Muestra la informacion de contacto"""
    help_text = (
        "*Puedes contactarnos:*\n\n"
        "**Telefono  📱**  ```0412\\-6413418```\n"
        "**Instagram Translanda  🚍** ```TransLanda\\.ca```\n"
        "**Instagram Promotora  🚍** ```Branyelislandaeta```\n"
        "**Correo  📩** ```Briam\\.landaeta@gmail\\.com```\n\n"
        "Puedes contactar con nosotros para cualquier información acerca de viajes disponibles, viajes futuros o servicio privado de transporte\\.\n" 
    )
    await update.message.reply_text(help_text, parse_mode="MarkdownV2")


# Buscar imagen en Google
def buscar_imagen(query):
    """Busca una imagen en Google Images."""
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_SEARCH_ENGINE_ID}&q={query}&searchType=image"
    response = requests.get(url)
    try:
        data = json.loads(response.text)
        if "items" in data:
            return data["items"][0]["link"]
    except json.JSONDecodeError:
        print("Error: La respuesta de la API no es un JSON válido.")
    return None


# Comando img (mostrar imagen)
async def img(update: Update, context: CallbackContext):
    """Maneja el comando /img"""
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Por favor, proporciona una búsqueda.")
        return
    image_url = buscar_imagen(query)
    if image_url:
        await context.bot.send_photo(chat_id=update.effective_chat.id,
                                     photo=image_url)
    else:
        await update.message.reply_text("No se encontró ninguna imagen.")


# Comando logo (muestra el logo)
async def mostrar_logo(update: Update, context: CallbackContext):
    """Muestra el logo con botones de respuesta"""
    keyboard = [[InlineKeyboardButton("Sí", callback_data="logo_si")],
                [InlineKeyboardButton("No", callback_data="logo_no")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_photo(chat_id=update.effective_chat.id,
                                 photo=LOGO_URL,
                                 caption="¿Te gusta nuestro logo?",
                                 reply_markup=reply_markup)


# Respuesta del logo
async def respuesta_logo(update: Update, context: CallbackContext):
    """Responde según la opción elegida en los botones"""
    query = update.callback_query
    await query.answer()

    if query.data == "logo_si":
        respuesta = "¡Nos alegra que te guste! 😊"
    else:
        respuesta = "Gracias por tu opinión. Trabajaremos en mejorar. "

    await query.edit_message_caption(
        caption=f"¿Te gusta el logo de nuestra empresa?\n\n{respuesta}")


async def dolar(update, context):  # Añadido 'async' aquí
    """Comando /dolar para mostrar el precio del dólar."""
    precio = obtener_precio_dolar()
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=precio)


def obtener_precio_dolar():
    """Obtiene el precio del dólar en bolívares desde Wise."""
    url = "https://wise.com/es/currency-converter/usd-to-ves-rate"

    try:
        # Realizar la solicitud a la página
        response = requests.get(url)
        response.raise_for_status()  # Verifica errores en la solicitud

        # Analizar el HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Buscar el valor del dólar en la estructura HTML
        precio_element = soup.find("span", class_="text-success")

        if precio_element:
            precio = precio_element.text.strip()
            return f"💵 1 USD = {precio} Bs. VES"
        else:
            return "⚠️ No se pudo encontrar el precio del dólar."

    except requests.exceptions.RequestException as e:
        return f"❌ Error al obtener el precio del dólar:https://www.bcv.org.ve/"


# Función para el comando /dolar
async def dolar(update: Update, context: CallbackContext):
    """Maneja el comando /dolar y envía el precio del dólar en bolívares"""
    precio_dolar = obtener_precio_dolar()
    await update.message.reply_text(precio_dolar)


# Diccionario
productos = {
    "producto1": {
        "nombre": "Adicora Full Day 🌊",
        "destino": "Adicora",
        "hora_s": "5:00 AM",
        "hora_r": "6:00 PM",
        "fecha": "01/02/2025",
        "precio": "$5",
        "metodos_pago_m":"Pago-Movil\n➖ Briam Landaeta\n➖ Banco: Venezuela\n➖ Cedula: 31580550\n➖ Telefono:0412-9558058",
        "metodos_pago_d":"Binance-Zelle-Paypal\n➖ Briam.Landaeta@gmail.com",
        "paradas":"Av.Intercomunal\n  ➖ Santa Clara\n  ➖ Barroso\n  ➖ 5 Bocas\n  ➖ Bolivia\n  ➖ Cumaná\n  ➖ La Norma\n  ➖ La Burbuja\n  ➖ La Estrella\n  ➖ La Cvp\n",
        "imagen": "img/adicora.jpg",
        "imagen_puestos": "img/imagen_puestos.png",
        "imagen_pago": "img/pago.png",
        "imagen_marca": "img/translanda.jpg",
        "imagen_mapa": "img/mapa.png",
    },
    "producto2": {
        "nombre": "Aquatica Full Day ⛱️",
        "destino": "Aquatica",
        "hora_s": "8:00 AM",
        "hora_r": "4:30 PM",
        "fecha": "18/04/2025",
        "precio": " $5",
        "metodos_pago_m":"Pago-Movil\n➖ Briam Landaeta\n➖ Banco: Venezuela\n➖ Cedula: 31580550\n➖ Telefono:0412-9558058",
        "metodos_pago_d":"Binance-Zelle-Paypal\n➖ Briam.Landaeta@gmail.com",
        "paradas":
        "Av.Intercomunal\n  ➖ Santa Clara\n  ➖ Barroso\n  ➖ 5 Bocas\n  ➖ Bolivia\n  ➖ Cumaná\n  ➖ La Norma\n  ➖ La Burbuja\n  ➖ La Estrella\n  ➖ La Cvp\n",
        "imagen": "img/aquatica.jpg",
        "imagen_puestos": "img/imagen_puestos.png",
        "imagen_pago": "img/pago.png",
        "imagen_marca": "img/translanda.jpg",
        "imagen_mapa": "img/mapa.png",
    }
}


async def show_productos(update: Update, context: CallbackContext) -> None:
    mensaje = "Estos son nuestros viajes disponibles selecione para mostrar mas detalles:\n\n"
    keyboard = []

    # Crear una fila con los botones de los productos
    keyboard.append([
        InlineKeyboardButton(f"Ver {producto['nombre']}",
                             callback_data=producto_id)
        for producto_id, producto in productos.items()
    ])

    # Usar una imagen local (marca_imagen)
    media = InputMediaPhoto(open(productos["producto1"]["imagen_marca"], 'rb'),
                            caption=mensaje)

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Verificar si es un callback_query o un mensaje nuevo
    if update.callback_query:
        try:
            await update.callback_query.edit_message_media(
                media=media)  # Edita el mensaje existente con la imagen
            await update.callback_query.edit_message_reply_markup(
                reply_markup=reply_markup)  # Edita los botones
        except Exception as e:
            print(f"Error editando mensaje: {e}")
            await update.callback_query.message.reply_text(
                text=mensaje, reply_markup=reply_markup, parse_mode="Markdown")
    elif update.message:
        await update.message.reply_photo(
            photo=open(productos["producto1"]["imagen_marca"],
                       'rb'),  # Usamos la imagen del producto
            caption=mensaje,
            reply_markup=reply_markup,
            parse_mode="Markdown")


async def show_detalles_producto(update: Update, context: CallbackContext) -> None:
    producto_id = update.callback_query.data
    producto = productos[producto_id]

    keyboard = [
        [
            InlineKeyboardButton("💺 Puestos Disponibles",
                                 callback_data=f"puestos_{producto_id}")
        ],  # Botón para Puestos disponibles
        [
            InlineKeyboardButton("⛔ Paradas Disponibles",
                                 callback_data=f"puntos_{producto_id}")
        ],  # Botón para Paradas disponibles
        [
            InlineKeyboardButton("💳 Métodos de Pago",
                                 callback_data=f"pago_{producto_id}")
        ],  # Botón de metodos de pago
        [InlineKeyboardButton("⬅️ Volver", callback_data="productos")]
    ]

    media = InputMediaPhoto(
        open(producto['imagen'], 'rb'),
        caption=
        f"\n\n\n`{producto['nombre']}`\n\n**Medio de transporte 🚌** ```Auto-Bus```\n**Horario**  ``` ⏰ Fecha del viaje:{producto['fecha']}\n ⏰ Hora de salida:{producto['hora_s']}\n ⏰ Hora retorno:{producto['hora_r']}``` **Destino 🚩** ```{producto['destino']}```\n **Precio P/P 💵** ```{producto['precio']}```\n\n**Cualquier tipo de duda, puedes comunicarte a nuestro WhatsApp, coloca /contacto para facilitarte los datos**",
        parse_mode="Markdown")

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_media(media=media)
    await update.callback_query.edit_message_reply_markup(
        reply_markup=reply_markup)


async def show_metodos_pago(update: Update, context: CallbackContext) -> None:
    producto_id = update.callback_query.data.split("_")[1]
    producto = productos.get(producto_id)

    if not producto:
        await update.callback_query.answer("❌ Producto no encontrado")
        return

    media = InputMediaPhoto(
        open(producto['imagen_pago'], 'rb'),
        caption=f"💳 *Métodos de pago:*\n```{producto['metodos_pago_m']}```\n```{producto['metodos_pago_d']}```\n\n**Antes de realizar un pago notificanos a nuestro WhatsApp, coloca /contacto para facilitarte los datos**",
        parse_mode="Markdown")

    keyboard = [[InlineKeyboardButton("⬅️ Volver", callback_data=producto_id)]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_media(media=media)
    await update.callback_query.edit_message_reply_markup(
        reply_markup=reply_markup)


async def show_puestos(update: Update, context: CallbackContext) -> None:
    producto_id = update.callback_query.data.split("_")[1]
    producto = productos.get(producto_id)

    if not producto:
        await update.callback_query.answer("❌ Producto no encontrado")
        return

    media = InputMediaPhoto(
        open(producto['imagen_puestos'], 'rb'),
        caption=
        f"```Puestos💺\nPustos Disponibles 🟢\nPustos Reservados 🔴```\nEn el momento de pagar tu pasaje podras selecionar tu puesto",
        parse_mode="Markdown")

    keyboard = [[InlineKeyboardButton("⬅️ Volver", callback_data=producto_id)]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_media(media=media)
    await update.callback_query.edit_message_reply_markup(
        reply_markup=reply_markup)


# Función para mostrar puntos de salida
async def show_paradas(update: Update, context: CallbackContext) -> None:
    producto_id = update.callback_query.data.split("_")[1]
    producto = productos.get(producto_id)

    if not producto:
        await update.callback_query.answer("❌ Producto no encontrado")
        return

    media = InputMediaPhoto(
        open(producto['imagen_mapa'], 'rb'),
        caption=f"⛔ *Paradas Disponibles:*\n\n```{producto['paradas']}```",
        parse_mode="Markdown")

    keyboard = [[InlineKeyboardButton("⬅️ Volver", callback_data=producto_id)]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.callback_query.edit_message_media(media=media)
    await update.callback_query.edit_message_reply_markup(
        reply_markup=reply_markup)


# Configurar aplicación de Telegram
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("comandos", comandos))
app.add_handler(CommandHandler("contacto", contacto))
app.add_handler(CommandHandler("img", img))
app.add_handler(CommandHandler("logo", mostrar_logo))
app.add_handler(CommandHandler("dolar", dolar))
app.add_handler(CommandHandler('viajes', show_productos))
app.add_handler(CallbackQueryHandler(show_productos, pattern="^productos$"))
app.add_handler(CallbackQueryHandler(show_detalles_producto, pattern="^(producto[0-9]+)$"))
app.add_handler(CallbackQueryHandler(show_metodos_pago, pattern="^pago_"))
app.add_handler(CallbackQueryHandler(show_puestos, pattern="^puestos_"))
app.add_handler(CallbackQueryHandler(show_paradas, pattern="^puntos_"))
app.add_handler(CallbackQueryHandler(respuesta_logo))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,handle_message))

web_app = Flask(__name__)


@web_app.route('/')
def home():
    return "Bot está activo!"


def keep_alive():

    def run():
        web_app.run(host='0.0.0.0', port=8080)

    server = Thread(target=run)
    server.daemon = True
    server.start()


if __name__ == "__main__":
    print("Bot iniciando...")
    keep_alive()
    print("Servidor web activo")
    app.run_polling(allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True)
