import pandas as pd # type: ignore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# === CONFIGURACIÓN ===
excel_path = "Resultados postulados.xlsx"  # Ruta del archivo Excel
sender_email = "convocatorias@fundacionmaradentro.cl"
app_password = "ciau fnfr nonu aatz"  # Reemplaza con tu contraseña de aplicación

# === LEE EL ARCHIVO EXCEL ===
df = pd.read_excel(excel_path)

# === FUNCIÓN PARA CREAR EL CUERPO DEL MENSAJE SEGÚN Estado ===
def generar_mensaje(Estado):
    if Estado.strip() == "Seleccionada":
        mensaje = """
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <p>Estimado/a,</p>

        <p>Nos alegra informarle que su proyecto ha pasado la <b>etapa de <b>preselección</b> de nuestro <b>Fondo FMA</b> ¡Felicitaciones!</p>

        <p>La siguiente etapa consistirá en una <b>entrevista</b> donde se buscará profundizar en las características y ejecución de su propuesta, a través de una <b>presentación online que Ud. podrá realizar junto a una persona más</b> si así lo estima. 
        La duración de la entrevista es de <b>15 minutos</b>: 10 para su presentación y 5 para preguntas de FMA.</p>

        <p><b>Características de la presentación:</b></p>
        <ul>
          <li>Considere dar cuenta de la definición del problema</li>
          <li>Describa cómo quieren aportar hacia una o más soluciones</li>
          <li>Explique la metodología a implementar</li>
          <li>Detalle con quiénes trabajarán</li>
          <li>Aborde qué esperan lograr</li>
        </ul>

        <p>Confirme <b>al menos dos opciones de horario en que podría participar de la entrevista</b>. 
        Recuerde que será un rango de 15 minutos. <b>Ejemplo</b>: <i>Miércoles 13 de noviembre 9 AM / Jueves 14 de noviembre 17.00 PM.</i></p>

        <p><b>Estos son los días/horas disponibles:</b></p>
        <ul>
          <li>Martes 12 de noviembre: 9.00 - 18.00 hrs</li>
          <li>Miércoles 13 de noviembre: 9.00 - 18.00 hrs</li>
          <li>Jueves 14 de noviembre: 9.00 - 18.00 hrs</li>
        </ul>

        <p>Luego de la instancia de entrevista se realizará un proceso final en el cual informaremos la lista definitiva de proyectos seleccionados/as.</p>

        <p>Mucha suerte en esta etapa y esperamos su respuesta.</p>

        <p>Saludos y felicidades nuevamente,
        <br><br>
        <b>Atte.<br>
        Equipo FMA</b></p>
        </body>
        </html>
        """
    else:
        mensaje = """
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
        <p>Estimado/a,</p>
        <br>
        <p>Buen día. Durante las últimas semanas, nuestro equipo ha estado revisando las postulaciones de la convocatoria <b>Fondo FMA 2025-2026</b>.</p>

        <p>Agradecemos sinceramente su esfuerzo y dedicación para participar de esta instancia concursable, 
        sin embargo, esta vez su propuesta no pasó al siguiente nivel de selección.</p>

        <p>Este año el nivel de las iniciativas fue especialmente alto, por lo que estamos muy agradecidos de conocer 
        el compromiso, la diversidad y el entusiasmo a lo largo de Chile para emprender acciones por el cuidado de la naturaleza.</p>

        <p>Debido a la gran cantidad de propuestas recibidas y la capacidad de respuesta de nuestro equipo, 
        no podremos entregar retroalimentación personalizada de cada proyecto.</p>

        <p>Esperamos hayan podido aprovechar las instancias de orientación, así como el material disponible 
        sobre proyectos seleccionados en versiones anteriores.</p>

        <p>En las próximas semanas publicaremos los resultados finales en nuestro sitio web y plataformas digitales.</p>

        <p>Esperamos que sus proyectos encuentren vías de ejecución y que puedan seguir atentos/as a nuestras convocatorias.</p>

        <p>Saludos, que tengan un excelente día.
        <br><br>
        <b>Atte.<br>
        Equipo FMA</b></p>
        </body>
        </html>
        """
    return mensaje

# === ENVÍO DE CORREOS ===
for _, row in df.iterrows():
    email_destino = str(row.get("Email", "")).strip()
    Estado = str(row.get("Estado", "")).strip()

    mensaje_html = generar_mensaje(Estado)

    # Crear correo
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = email_destino
    msg["Subject"] = "ASUNTO: Aviso proceso de selección Fondo FMA 2025-2026"

    # Agregar cuerpo HTML
    msg.attach(MIMEText(mensaje_html, "html"))

    # Enviar por Gmail
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

    print(f"✅ Enviado a {email_destino} — Estado: {Estado}")
    time.sleep(2)  # ⏸️ small pause to prevent Gmail throttling
    
print("🎉 Todos los correos fueron enviados correctamente.")
