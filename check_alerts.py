import os
from dotenv import load_dotenv
from plyer import notification
import smtplib, ssl
from email.mime.text import MIMEText
import numpy as np
import base64
import webbrowser

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")


def play_alarm():
    mp3_path="siren.mp3"
    os.system(f"start {mp3_path}")


def send_email_alert(region_id, prob):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    TO_EMAIL = os.getenv("EMAIL_TO", EMAIL_USER)

    if not EMAIL_USER or not EMAIL_PASS:
        print("Email credentials missing!")
        return

    subject = f"Wildfire Risk ALERT — Region {region_id}"
    body = f"""
High wildfire risk detected!

Region: {region_id}
Risk Probability: {prob:.2f}

Stay alert & take precautions!
"""

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = TO_EMAIL

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, TO_EMAIL, msg.as_string())
        server.quit()

        print(f"Test email sent to {TO_EMAIL} for region {region_id} (prob={prob:.2f})")

    except Exception as e:
        print(f"Email failed: {e}")

def send_windows_alert(region_id, prob):
    try:
        notification.notify(
            title=f"Fire Risk Alert — Region {region_id}",
            message=f"Probability: {prob:.2f}",
            timeout=7
        )
        print(f"Windows alert sent for region {region_id}")
    except:
        print("Windows alert failed (desktop only)")

if __name__ == "__main__":
    region_id = 14
    prob = 0.93

    print("Playing alarm!!!!")
    play_alarm()

    print("Sending email alert!!!!")
    send_email_alert(region_id, prob)

    print("Sending Windows popup!!!!!!")
    send_windows_alert(region_id, prob)
