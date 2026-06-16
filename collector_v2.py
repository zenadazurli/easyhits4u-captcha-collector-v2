#!/usr/bin/env python3
# collector_v2.py
# Raccoglie captcha non risolti (figure e matematici) usando i cookie

import os
import sys
import time
import random
import requests
import json
import re
import numpy as np
import cv2
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3
from PIL import Image
import io

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIGURAZIONE ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "easyhits4u-captchas-v2"
DATASET_REPO = "zenadazurli/easyhits4u-dataset"
DIM = 64
REQUEST_TIMEOUT = 15

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL e SUPABASE_KEY devono essere impostate")

# ==================== INIZIALIZZAZIONE ====================
X_fast = None
y_fast = None
classes_fast = None

class CaptchaCollectorV2:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session = requests.Session()
        self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0, 'risolti': 0}
        self.account_name = None
        self.current_cookie = None
        
        # Carica il dataset una volta all'avvio
        self.load_dataset()
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    def load_dataset(self):
        """Carica il dataset delle figure da Hugging Face"""
        global X_fast, y_fast, classes_fast
        self.log(f"📥 Caricamento dataset da Hugging Face: {DATASET_REPO}")
        
        try:
            dataset = load_dataset(DATASET_REPO, trust_remote_code=True)
            data = dataset.get("train") if "train" in dataset else dataset
            
            X = []
            y = []
            class_to_idx = {}
            
            for item in data:
                features = item.get("X")
                label_idx = item.get("y")
                if features is None or label_idx is None:
                    continue
                
                if hasattr(data.features['y'], 'names'):
                    class_name = data.features['y'].names[label_idx]
                else:
                    class_name = str(label_idx)
                
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(class_to_idx)
                
                X.append(np.array(features, dtype=np.float32))
                y.append(class_to_idx[class_name])
            
            if X:
                X_fast = np.vstack(X).astype(np.float32)
                y_fast = np.array(y, dtype=np.int32)
                classes_fast = {v: k for k, v in class_to_idx.items()}
                self.log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi")
                return True
            else:
                self.log("❌ Nessun dato valido nel dataset")
                return False
        except Exception as e:
            self.log(f"❌ Errore caricamento dataset: {e}")
            return False
    
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
    
    # ==================== FUNZIONI PER FIGURE ====================
    def centra_figura(self, image):
        """Centra e ritaglia la figura"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(image, (DIM, DIM))
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y+h, x:x+w]
        return cv2.resize(crop, (DIM, DIM))
    
    def estrai_descrittori(self, img):
        """Estrae descrittori per la figura"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circularity = 0.0
        aspect_ratio = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if peri != 0:
                circularity = 4.0 * np.pi * area / (peri * peri)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h if h != 0 else 0.0
        
        moments = cv2.moments(thresh)
        hu = cv2.HuMoments(moments).flatten().tolist()
        
        h, w = img.shape[:2]
        cx, cy = w//2, h//2
        raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
        radiale = []
        for r in raggi:
            mask = np.zeros((h,w), np.uint8)
            cv2.circle(mask, (cx,cy), r, 255, -1)
            mean = cv2.mean(img, mask=mask)[:3]
            radiale.extend([m/255.0 for m in mean])
        
        spaziale = []
        quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
        for (x1,y1,x2,y2) in quadranti:
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                mean = cv2.mean(roi)[:3]
                spaziale.extend([m/255.0 for m in mean])
        
        return radiale + spaziale + [circularity, aspect_ratio] + hu
    
    def predict_figure(self, img_crop):
        """Riconosce una figura usando il dataset"""
        global X_fast, y_fast, classes_fast
        
        if X_fast is None or img_crop is None or img_crop.size == 0:
            return None
        
        img_centrata = self.centra_figura(img_crop)
        features = np.array(self.estrai_descrittori(img_centrata), dtype=float)
        distances = np.linalg.norm(X_fast - features, axis=1)
        best_idx = np.argmin(distances)
        return classes_fast.get(int(y_fast[best_idx]), None)
    
    def crop_safe(self, img, coords):
        """Ritaglia in sicurezza dalle coordinate"""
        try:
            x1, y1, x2, y2 = map(int, coords.split(","))
        except:
            return None
        h, w = img.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]
    
    def risolvi_figure(self, urlid, qpic, picmap, img):
        """Risolve captcha a figure"""
        self.log("   🖼️ Captcha a figure rilevato")
        
        # Ritaglia le 5 immagini
        crops = [self.crop_safe(img, p.get("coords", "")) for p in picmap]
        
        # Riconosci ogni figura
        labels = []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                label = self.predict_figure(crop)
                labels.append(label)
                self.log(f"      Figura {i+1}: {label}")
            else:
                labels.append(None)
                self.log(f"      Figura {i+1}: errore")
        
        # Cerca duplicati
        seen = {}
        chosen_idx = None
        for i, label in enumerate(labels):
            if label and label != "errore":
                if label in seen:
                    chosen_idx = seen[label]
                    break
                seen[label] = i
        
        if chosen_idx is None:
            self.log("   ❌ Nessun duplicato trovato")
            self.salva_captcha(qpic, img, picmap, labels, "nessun_duplicato", urlid)
            return None
        
        # Invia la risposta
        word = picmap[chosen_idx]["value"]
        self.log(f"   ✅ Duplicato: figura {chosen_idx+1} -> word={word}")
        return word
    
    # ==================== FUNZIONI PER MATEMATICI ====================
    def risolvi_matematico(self, urlid, surfses):
        """Risolve captcha matematico"""
        self.log("   🧮 Captcha matematico rilevato")
        
        # Estrai i numeri
        aword1 = surfses.get("aword1")
        aword2 = surfses.get("aword2")
        aword3 = surfses.get("aword3")
        num1 = surfses.get("aword1_number")
        num2 = surfses.get("aword2_number")
        num3 = surfses.get("aword3_number")
        
        self.log(f"      Numeri: {num1}, {num2}, {num3}")
        self.log(f"      Parole: {aword1}, {aword2}, {aword3}")
        
        # Al momento salviamo sempre i matematici perché non abbiamo risolutore
        self.log("   ⚠️ Captcha matematico - SALVO PER ANALISI")
        # Qui salveremo l'immagine e i dati
        return None
    
    # ==================== FUNZIONI COMUNI ====================
    def salva_captcha(self, qpic, img, picmap, labels, motivo, urlid=None):
        """Salva il captcha non risolto su Supabase"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            
            # Determina tipo
            if picmap is not None:
                prefix = "figure"
                table = "figure_captchas_v2"
                tipo = "figure"
            else:
                prefix = "math"
                table = "math_captchas_v2"
                tipo = "math"
            
            file_path = f"{prefix}/{timestamp}_{self.account_name}.png"
            _, buffer = cv2.imencode('.png', img)
            img_bytes = buffer.tobytes()
            
            self.supabase.storage.from_(BUCKET_NAME).upload(file_path, img_bytes)
            
            data = {
                'account_name': self.account_name,
                'image_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'unsolved',
                'motivo': motivo,
                'urlid': urlid,
                'qpic': qpic
            }
            
            if labels:
                data['labels_predette'] = json.dumps(labels)
            
            self.supabase.table(table).insert(data).execute()
            
            if tipo == "figure":
                self.stats['figure'] += 1
            else:
                self.stats['math'] += 1
            
            self.log(f"   💾 Captcha salvato ({tipo})")
            return True
        except Exception as e:
            self.log(f"   ❌ Errore salvataggio: {e}")
            return False
    
    def surf_and_get_captcha(self):
        """Esegue un ciclo di surf e restituisce il captcha"""
        try:
            r = self.session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            
            if r.status_code != 200:
                return None, None, "errore_http"
            
            data = r.json()
            surfses = data.get("surfses", {})
            urlid = surfses.get("urlid")
            qpic = surfses.get("qpic")
            picmap = data.get("picmap")
            
            if not urlid or not qpic:
                return None, None, "cookie_scaduto"
            
            # Scarica l'immagine
            img_data = self.session.get(
                f"https://www.easyhits4u.com/simg/{qpic}.jpg",
                verify=False
            ).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            return {
                'data': data,
                'urlid': urlid,
                'qpic': qpic,
                'picmap': picmap,
                'surfses': surfses,
                'img': img
            }, None, "ok"
            
        except Exception as e:
            return None, str(e), "errore"
    
    def invia_risposta(self, urlid, word):
        """Invia la risposta al captcha"""
        try:
            url = f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2&ajax=1&word={word}&screen_width=1024&screen_height=768"
            
            # Aggiungi i parametri mancanti
            url += "&window_width=1024&window_height=643&top_width=1024&top_height=50"
            url += "&fpcode=TW96aWxsYTsgTmV0c2NhcGU7IDUuMCAoV2luZG93cyk7IFdpbjMy"
            url += f"&cit={int(time.time() * 1000)}&try=1"
            
            r = self.session.get(url, verify=False, timeout=REQUEST_TIMEOUT)
            return r.json(), "ok"
        except Exception as e:
            return None, str(e)
    
    def run_account(self, account_name, sesids):
        """Esegue surf per un account usando il cookie"""
        self.account_name = account_name
        self.current_cookie = sesids
        
        self.log(f"📧 Account: {account_name}")
        self.session.cookies.set('sesids', sesids)
        
        for i in range(10):  # 10 tentativi di surf
            result, error, status = self.surf_and_get_captcha()
            
            if status == "cookie_scaduto":
                self.log(f"   ❌ Cookie scaduto per {account_name}")
                return
            
            if status != "ok" or result is None:
                self.log(f"   ⚠️ Errore surf: {status}")
                continue
            
            # Determina il tipo di captcha
            if result['picmap'] is not None:
                # FIGURE
                word = self.risolvi_figure(
                    result['urlid'],
                    result['qpic'],
                    result['picmap'],
                    result['img']
                )
            else:
                # MATEMATICO
                word = self.risolvi_matematico(
                    result['urlid'],
                    result['surfses']
                )
                # Se non abbiamo risolutore, salviamo e fermiamo
                if word is None:
                    self.salva_captcha(
                        result['qpic'],
                        result['img'],
                        None,
                        None,
                        "matematico_non_risolto",
                        result['urlid']
                    )
                    return
            
            if word is None:
                # Non risolto - ferma account
                return
            
            # Invia risposta
            response, status_invio = self.invia_risposta(result['urlid'], word)
            
            if response and response.get("warning") == "wrong_choice":
                self.log(f"   ❌ Risposta sbagliata: {word}")
                # Salva l'errore
                self.salva_captcha(
                    result['qpic'],
                    result['img'],
                    result['picmap'],
                    None,
                    "wrong_choice",
                    result['urlid']
                )
                return
            
            self.stats['risolti'] += 1
            self.log(f"   ✅ OK #{self.stats['risolti']} - word={word}")
            
            time.sleep(random.uniform(2, 4))
        
        self.log(f"   ✅ Completato: {self.stats['risolti']} risolti")
    
    def run(self):
        self.log("=" * 60)
        self.log("🚀 CAPTCHA COLLECTOR V2 - CONFIGURAZIONE COMPLETA")
        self.log("=" * 60)
        
        cookies = self.get_cookies_from_supabase()
        if not cookies:
            self.log("❌ Nessun cookie trovato")
            return
        
        self.log(f"📋 Cookie disponibili: {len(cookies)}")
        
        for account_name, sesids in cookies.items():
            # Reset stats per account
            self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0, 'risolti': 0}
            self.run_account(account_name, sesids)
            time.sleep(random.uniform(3, 6))
        
        self.log("=" * 60)
        self.log("📊 STATISTICHE TOTALI")
        self.log(f"   Figure salvate: {self.stats['figure']}")
        self.log(f"   Matematici salvati: {self.stats['math']}")
        self.log(f"   Captcha risolti: {self.stats['risolti']}")
        self.log("=" * 60)

if __name__ == "__main__":
    collector = CaptchaCollectorV2()
    collector.run()
