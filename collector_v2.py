#!/usr/bin/env python3
# collector_v2.py

import os
import time
import random
import requests
import re
from datetime import datetime
from supabase import create_client, Client
import cv2
import numpy as np
from PIL import Image
import io
import json

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "easyhits4u-captchas-v2"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL e SUPABASE_KEY devono essere impostate")

def carica_accounts():
    accounts = []
    if not os.path.exists("accounts.txt"):
        raise FileNotFoundError("❌ File accounts.txt non trovato")
    with open("accounts.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(",")
                if len(parts) >= 2:
                    accounts.append((parts[0].strip(), parts[1].strip()))
    return accounts

class CaptchaCollectorV2:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session = requests.Session()
        self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0}
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    def login(self, email, password):
        try:
            login_url = "https://www.easyhits4u.com/logon/"
            self.session.get(login_url)
            data = {"username": email, "password": password, "submit": "Sign In"}
            self.session.post(login_url, data=data)
            return "sesids" in [c.name for c in self.session.cookies]
        except Exception as e:
            self.log(f"Login error: {e}")
            return False
    
    def download_captcha(self):
        try:
            surf_url = "https://www.easyhits4u.com/surf/"
            response = self.session.get(surf_url)
            patterns = [
                r'src="([^"]+captcha[^"]+)"',
                r'src="([^"]+Captcha[^"]+)"',
            ]
            captcha_url = None
            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    captcha_url = match.group(1)
                    break
            if not captcha_url:
                return None, None
            if not captcha_url.startswith("http"):
                captcha_url = "https://www.easyhits4u.com" + captcha_url
            img_response = self.session.get(captcha_url)
            is_math = "math" in captcha_url.lower() or "numeric" in captcha_url.lower()
            return img_response.content, is_math
        except Exception as e:
            self.log(f"Download error: {e}")
            return None, None
    
    def salva_su_supabase(self, account_name, img_bytes, is_math):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            prefix = "math" if is_math else "figure"
            file_path = f"{prefix}/{timestamp}_{account_name}.png"
            self.supabase.storage.from_(BUCKET_NAME).upload(file_path, img_bytes)
            table = "math_captchas_v2" if is_math else "figure_captchas_v2"
            data = {
                'account_name': account_name,
                'image_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'unsolved'
            }
            self.supabase.table(table).insert(data).execute()
            return True
        except Exception as e:
            self.log(f"Save error: {e}")
            return False
    
    def run_account(self, account):
        email, password = account
        name = email.split('+')[1].split('@')[0] if '+' in email else email.split('@')[0]
        self.log(f"📧 Account: {name}")
        if not self.login(email, password):
            self.log(f"   ❌ Login fallito")
            self.stats['errors'] += 1
            return
        self.log(f"   ✅ Login riuscito")
        for i in range(10):
            img_bytes, is_math = self.download_captcha()
            if img_bytes:
                if self.salva_su_supabase(name, img_bytes, is_math):
                    if is_math:
                        self.stats['math'] += 1
                        self.log(f"   📊 Captcha matematico #{i+1} salvato")
                    else:
                        self.stats['figure'] += 1
                        self.log(f"   🖼️ Captcha figure #{i+1} salvato")
            else:
                self.log(f"   ⚠️ Nessun captcha trovato")
            self.stats['surf'] += 1
            time.sleep(random.uniform(2, 5))
    
    def run(self):
        self.log("=" * 60)
        self.log("🚀 CAPTCHA COLLECTOR V2 - 40 NUOVI ACCOUNT")
        self.log("=" * 60)
        accounts = carica_accounts()
        self.log(f"📋 Caricati {len(accounts)} account")
        for account in accounts:
            self.run_account(account)
            time.sleep(random.uniform(3, 7))
        self.log("=" * 60)
        self.log("📊 STATISTICHE FINALI")
        self.log(f"   Surf: {self.stats['surf']}")
        self.log(f"   Figure salvate: {self.stats['figure']}")
        self.log(f"   Matematici salvati: {self.stats['math']}")
        self.log(f"   Errori: {self.stats['errors']}")
        self.log("=" * 60)

if __name__ == "__main__":
    collector = CaptchaCollectorV2()
    collector.run()