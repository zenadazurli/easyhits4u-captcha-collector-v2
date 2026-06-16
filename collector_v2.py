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

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "easyhits4u-captchas-v2"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL e SUPABASE_KEY devono essere impostate")

class CaptchaCollectorV2:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session = requests.Session()
        self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0}
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    def get_cookies_from_supabase(self):
        """Legge i cookie dalla tabella account_cookies"""
        try:
            result = self.supabase.table('account_cookies').select('account_name, sesids').execute()
            cookies = {}
            for row in result.data:
                if row.get('sesids'):
                    cookies[row['account_name']] = row['sesids']
            self.log(f"📋 Letti {len(cookies)} cookie da Supabase")
            return cookies
        except Exception as e:
            self.log(f"❌ Errore lettura cookie: {e}")
            return {}
    
    def download_captcha(self):
        try:
            surf_url = "https://www.easyhits4u.com/surf/"
            response = self.session.get(surf_url)
            
            if "logon" in response.url.lower() or "login" in response.url.lower():
                return None, None, "cookie_invalido"
            
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
                return None, None, "nessun_captcha"
            
            if not captcha_url.startswith("http"):
                captcha_url = "https://www.easyhits4u.com" + captcha_url
            
            img_response = self.session.get(captcha_url)
            is_math = "math" in captcha_url.lower() or "numeric" in captcha_url.lower()
            
            return img_response.content, is_math, "ok"
        except Exception as e:
            self.log(f"Download error: {e}")
            return None, None, "errore"
    
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
    
    def run_account(self, account_name, sesids):
        self.log(f"📧 Account: {account_name}")
        
        self.session.cookies.set('sesids', sesids)
        
        surf_count = 0
        for i in range(8):
            img_bytes, is_math, status = self.download_captcha()
            
            if status == "cookie_invalido":
                self.log(f"   ❌ Cookie non valido per {account_name}")
                return
            
            if img_bytes:
                if self.salva_su_supabase(account_name, img_bytes, is_math):
                    if is_math:
                        self.stats['math'] += 1
                        self.log(f"   📊 Captcha matematico #{i+1} salvato")
                    else:
                        self.stats['figure'] += 1
                        self.log(f"   🖼️ Captcha figure #{i+1} salvato")
                surf_count += 1
            else:
                self.log(f"   ⚠️ Nessun captcha trovato (status: {status})")
            
            self.stats['surf'] += 1
            time.sleep(random.uniform(3, 6))
        
        self.log(f"   ✅ Completato: {surf_count} surf")
    
    def run(self):
        self.log("=" * 60)
        self.log("🚀 CAPTCHA COLLECTOR V2 - 40 NUOVI ACCOUNT")
        self.log("=" * 60)
        
        cookies = self.get_cookies_from_supabase()
        
        if not cookies:
            self.log("❌ Nessun cookie trovato in Supabase")
            return
        
        self.log(f"📋 Cookie disponibili: {len(cookies)}")
        
        for account_name, sesids in cookies.items():
            self.run_account(account_name, sesids)
            time.sleep(random.uniform(2, 5))
        
        self.log("=" * 60)
        self.log("📊 STATISTICHE FINALI")
        self.log(f"   Surf effettuati: {self.stats['surf']}")
        self.log(f"   Figure salvate: {self.stats['figure']}")
        self.log(f"   Matematici salvati: {self.stats['math']}")
        self.log(f"   Errori: {self.stats['errors']}")
        self.log("=" * 60)

if __name__ == "__main__":
    collector = CaptchaCollectorV2()
    collector.run()
