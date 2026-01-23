import smtplib, ssl

email = ""        # 👈 your Gmail address here
app_password = ""     # 👈 your 16-char app password (NO spaces)

try:
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(email, app_password)
        print("✅ Login successful! Your app password works correctly.")
except Exception as e:
    print("❌ Login failed. Error details:")
    print(e)
