#!/usr/bin/env python3
# collector_v2.py
# Basato sul repository easyhits4u-surf-collector-main
# USA SOLO I 40 NUOVI ACCOUNT (nessun account vecchio)

import os
import sys
import time
import random
import requests
import json
import numpy as np
import cv2
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3

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

# ==================== LISTA DEI 40 NUOVI ACCOUNT ====================
NUOVI_ACCOUNT = [
    'ucupamikowa', 'ubbmaad', 'unachizaadaa', 'uzofequ',
    'ugaglchimulu', 'usfnejafi', 'ugaufkokagl', 'utuufvo',
    'umufela', 'uzukimice', 'uvatulukofo', 'ugetrle',
    'usfkugl', 'uzuculo', 'uxipgda', 'ulidazurzmu',
    'uncglximo', 'ufezusavo', 'ulileaature', 'ulorenakino',
    'uqulenazusa', 'ukaramu', 'uferalola', 'ummmarzsarm',
    'udatrlefe', 'uaakiggzu', 'uzorzvu', 'uwanepgbo',
    'udioodali', 'usadiadmobo', 'ulixire', 'udiadnczo',
    'uzalesagg', 'upabbkafone', 'uramincadkr', 'uganakaeara',
    'urerafokrne', 'ufiwakota', 'ukrfojudi', 'uornewafomo'
]

# ==================== VARIABILI GLOBALI ====================
X_fast = None
y_fast = None
classes_fast = None

# ==================== FUNZIONI DATASET ====================
def load_dataset_from_hf():
    """Carica il dataset da Hugging Face"""
    global X_fast, y_fast, classes_fast
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 Caricamento dataset da Hugging Face: {DATASET_REPO}", flush=True)
    
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi", flush=True)
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Nessun dato valido nel dataset", flush=True)
            return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Errore caricamento dataset: {e}", flush=True)
        return False

# ==================== FUNZIONI FIGURE ====================
def centra_figura(image):
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
    crop = image[y:y+h, x:x+w)
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
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

def predict_figure(img_crop):
    """Riconosce una figura usando il dataset"""
    global X_fast, y_fast, classes_fast
    
    if X_fast is None or img_crop is None or img_crop.size == 0:
        return None
    
    img_centrata = centra_figura(img_crop)
    features = np.array(estrai_descrittori(img_centrata), dtype=float)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), None)

def crop_safe(img, coords):
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

# ==================== COLLECTOR ====================
class CaptchaCollectorV2:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session = requests.Session()
        self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0, 'risolti': 0}
        self.account_name = None
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)
    
    def get_cookies_from_supabase(self):
        """Legge SOLO i cookie dei 40 nuovi account"""
        try:
            result = self.supabase.table('account_cookies')\
                .select('account_name, cookie_string, sesids')\
                .in_('account_name', NUOVI_ACCOUNT)\
                .execute()
            
            cookies = {}
            for row in result.data:
                cookie = row.get('cookie_string')
                if not cookie and row.get('sesids'):
                    cookie = f"sesids={row['sesids']}"
                if cookie:
                    cookies[row['account_name']] = cookie
            
            self.log(f"📋 Letti {len(cookies)} cookie dei 40 nuovi account")
            return cookies
        except Exception as e:
            self.log(f"❌ Errore lettura cookie: {e}")
            return {}
    
    def salva_captcha(self, qpic, img, picmap, labels, motivo, urlid=None):
        """Salva il captcha non risolto su Supabase"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            
            if picmap is not None:
                prefix = "figure"
                table = "figure_captchas_v2"
            else:
                prefix = "math"
                table = "math_captchas_v2"
            
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
            
            if picmap is not None:
                self.stats['figure'] += 1
            else:
                self.stats['math'] += 1
            
            self.log(f"   💾 Captcha salvato")
            return True
        except Exception as e:
            self.log(f"   ❌ Errore salvataggio: {e}")
            return False
    
    def risolvi_figure(self, urlid, qpic, picmap, img):
        """Risolve captcha a figure (come nell'originale)"""
        self.log("   🖼️ Captcha a figure rilevato")
        
        # Ritaglia le 5 immagini
        crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
        
        # Riconosci ogni figura
        labels = []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                label = predict_figure(crop)
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
        
        word = picmap[chosen_idx]["value"]
        self.log(f"   ✅ Duplicato: figura {chosen_idx+1} -> word={word}")
        return word
    
    def risolvi_matematico(self, urlid, surfses, img):
        """Salva captcha matematico per analisi"""
        self.log("   🧮 Captcha matematico rilevato - SALVO PER ANALISI")
        self.salva_captcha(
            surfses.get("qpic"),
            img,
            None,
            None,
            "matematico_non_risolto",
            urlid
        )
        return None
    
    def surf_and_get_captcha(self):
        """Esegue un ciclo di surf (come nell'originale)"""
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
                return None, None, "nessun_captcha"
            
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
        """Invia la risposta (come nell'originale)"""
        try:
            url = f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2&ajax=1&word={word}&screen_width=1024&screen_height=768"
            url += "&window_width=1024&window_height=643&top_width=1024&top_height=50"
            url += "&fpcode=TW96aWxsYTsgTmV0c2NhcGU7IDUuMCAoV2luZG93cyk7IFdpbjMy"
            url += f"&cit={int(time.time() * 1000)}&try=1"
            
            r = self.session.get(url, verify=False, timeout=REQUEST_TIMEOUT)
            return r.json(), "ok"
        except Exception as e:
            return None, str(e)
    
    def run_account(self, account_name, cookie_string):
        """Esegue surf per un account usando il cookie completo"""
        self.account_name = account_name
        
        # Imposta il cookie completo
        self.session.headers.update({"Cookie": cookie_string})
        
        self.log(f"📧 Account: {account_name}")
        
        # Attiva la sessione di surf
        try:
            self.log(f"   🔄 Attivazione sessione surf...")
            self.session.get("https://www.easyhits4u.com/surf/", verify=False, timeout=10)
            time.sleep(2)
        except Exception as e:
            self.log(f"   ⚠️ Errore attivazione surf: {e}")
        
        errori_consecutivi = 0
        MAX_ERRORI = 3
        captcha_counter = 0
        
        for i in range(15):
            result, error, status = self.surf_and_get_captcha()
            
            if status == "nessun_captcha":
                self.log(f"   ⚠️ Nessun captcha trovato")
                errori_consecutivi += 1
                if errori_consecutivi >= MAX_ERRORI:
                    self.log(f"   ❌ Troppi errori, fermo account")
                    return
                time.sleep(3)
                continue
            
            if status != "ok" or result is None:
                self.log(f"   ⚠️ Errore: {status}")
                errori_consecutivi += 1
                if errori_consecutivi >= MAX_ERRORI:
                    self.log(f"   ❌ Troppi errori, fermo account")
                    return
                continue
            
            errori_consecutivi = 0
            
            if result['picmap'] is not None:
                word = self.risolvi_figure(
                    result['urlid'],
                    result['qpic'],
                    result['picmap'],
                    result['img']
                )
            else:
                word = self.risolvi_matematico(
                    result['urlid'],
                    result['surfses'],
                    result['img']
                )
                return
            
            if word is None:
                return
            
            response, status_invio = self.invia_risposta(result['urlid'], word)
            
            if response and response.get("warning") == "wrong_choice":
                self.log(f"   ❌ Risposta sbagliata: {word}")
                self.salva_captcha(
                    result['qpic'],
                    result['img'],
                    result['picmap'],
                    None,
                    "wrong_choice",
                    result['urlid']
                )
                return
            
            captcha_counter += 1
            self.stats['risolti'] += 1
            self.log(f"   ✅ OK #{captcha_counter} - word={word}")
            
            time.sleep(random.uniform(2, 4))
        
        self.log(f"   ✅ Completato: {captcha_counter} captcha risolti")
    
    def run(self):
        self.log("=" * 60)
        self.log("🚀 CAPTCHA COLLECTOR V2 - 40 NUOVI ACCOUNT")
        self.log("=" * 60)
        
        if not load_dataset_from_hf():
            self.log("❌ Impossibile proseguire senza dataset")
            return
        
        cookies = self.get_cookies_from_supabase()
        if not cookies:
            self.log("❌ Nessun cookie trovato per i 40 nuovi account")
            return
        
        self.log(f"📋 Cookie disponibili: {len(cookies)}")
        
        for account_name, cookie_string in cookies.items():
            self.stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0, 'risolti': 0}
            self.run_account(account_name, cookie_string)
            time.sleep(random.uniform(2, 5))
        
        self.log("=" * 60)
        self.log("📊 STATISTICHE TOTALI")
        self.log(f"   Figure salvate: {self.stats['figure']}")
        self.log(f"   Matematici salvati: {self.stats['math']}")
        self.log(f"   Captcha risolti: {self.stats['risolti']}")
        self.log("=" * 60)

if __name__ == "__main__":
    collector = CaptchaCollectorV2()
    collector.run()
