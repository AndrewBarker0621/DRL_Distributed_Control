"""Mail notification stub: the original sends an e-mail at the end of a training run.
Set MAIL_APP_PASSWORD / MAIL_TO in the environment if you want that; otherwise this is a no-op."""
import os


def send_email_notification(subject, body, to_email=None):
    if not os.environ.get("MAIL_APP_PASSWORD"):
        return
    import smtplib
    from email.mime.text import MIMEText
    to_email = to_email or os.environ.get("MAIL_TO")
    msg = MIMEText(body); msg["Subject"] = subject; msg["To"] = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(os.environ["MAIL_USER"], os.environ["MAIL_APP_PASSWORD"]); s.send_message(msg)
