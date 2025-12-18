import time, os
from urllib.parse import urlparse
import requests
import qrcode
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, redirect, url_for

# 导入自定义模块
from app.getHistoryData import get_history_data
from app.getHotData import get_hot_data
from app.getRecommandData import get_recommand_data
from app.Recommender import Recommender

# app = Flask(__name__)

# --- 常量定义 ---
QR_CODE_GENERATE_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
QR_CODE_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}

# --- 全局变量 ---
cookie_file_path = "user_data/cookie.txt"
cookie_data = {}
cookie_str = ""

# 不再需要图片保存路径，因为我们使用远程链接
# img_path = "app/static/user_img"


recommender = None


# --- 工具函数 ---

def get_qrcodekey():
    try:
        response = requests.get(QR_CODE_GENERATE_URL, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["code"] == 0:
                return data["data"]["url"] + "main-fe-header", data["data"]["qrcode_key"]
    except Exception as e:
        print(f"获取二维码失败: {e}")
    return None, None


def generate_qrcode_base64(_url):
    if _url:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(_url)
        qr.make(fit=True)
        img = qr.make_image(fill="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    return None


def check_qrcode_status(qrcode_key):
    global cookie_data, cookie_str
    try:
        response = requests.get(QR_CODE_POLL_URL, params={"qrcode_key": qrcode_key}, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            code = data["data"]["code"]
            if code == 0:
                cookie_data = response.cookies.get_dict()
                cookie_str = "; ".join([f"{key}={value}" for key, value in cookie_data.items()])
                cookie_str = "buvid3=1; " + cookie_str
                timestamp = data["data"]["timestamp"]
                url = data["data"]["url"]
                return code, timestamp, url, cookie_str
            else:
                return code, None, None, None
    except Exception as e:
        print(f"检查二维码状态失败: {e}")
    return None, None, None, None


def get_history_info():
    try:
        # 直接获取数据，不再下载图片
        history_info = get_history_data(cookie_str, 12)
    except:
        history_info = []
    return history_info


# --- 路由函数 ---

def home():
    global cookie_str, headers
    if os.path.exists(cookie_file_path):
        with open(cookie_file_path, "r", encoding="utf-8") as f:
            cookie_str = f.read().strip()
            headers["Cookie"] = cookie_str
            return dashboard()
    else:
        return login()


def qrcode_status():
    global headers, recommender
    qrcode_key = request.args.get("qrcode_key")
    if not qrcode_key:
        return jsonify({"error": "Missing qrcode_key"}), 400

    status, timestamp, url, cookies = check_qrcode_status(qrcode_key)
    if status == 0:
        print("登录成功")
        # 重置模型
        recommender = None

        dir_path = os.path.dirname(cookie_file_path)
        if not os.path.exists(dir_path): os.makedirs(dir_path)

        with open(cookie_file_path, "w", encoding="utf-8") as f:
            f.write(f"{cookies}")
        headers["Cookie"] = cookies

        return jsonify({
            "status": "login success",
            "timestamp": timestamp,
            "url": url,
            "cookies": cookies,
        }), 200

    msg = "unknown"
    if status == 86101:
        msg = "not scanned"
    elif status == 86038:
        msg = "expired"
    elif status == 86090:
        msg = "scanned"
    return jsonify({"status": msg}), 200 if status != 86038 else 400


def login():
    url, qrcode_key = get_qrcodekey()
    qr_base64 = generate_qrcode_base64(url)
    if url and qrcode_key:
        return render_template("login.html", qr_code=qr_base64, qrcode_key=qrcode_key)
    return "Error generating QR code."


def dashboard():
    global cookie_str, recommender
    headers["Cookie"] = cookie_str

    if recommender is None and cookie_str:
        print("Dashboard 加载中...")

    history_info = get_history_info()

    #  只做简单的标签截断，不做图片路径替换
    for x in history_info:
        temp = []
        count = 0
        tags = x.get("tag", [])
        if tags:
            for xx in tags:
                if len(str(xx)) <= 8 and count < 2:
                    temp.append(xx)
                    count += 1
        x["tag"] = temp

    return render_template("dashboard.html", cookie_str=cookie_str, history_info=history_info)


def logout():
    global recommender
    if os.path.exists(cookie_file_path):
        os.remove(cookie_file_path)
    recommender = None
    return jsonify({"success": True})


# --- 核心推荐接口 ---

def recommend_hot_vid():
    global recommender, cookie_str

    if recommender is None:
        print("初始化推荐模型...")
        try:
            recommender = Recommender(cookie_str)
        except Exception as e:
            print(f"模型初始化失败: {e}")
            return jsonify([])
    else:
        recommender.cookies = cookie_str

    print("计算热门推荐...")
    start_t = time.time()
    try:
        res = recommender.recommend("hot", 8)
    except Exception as e:
        print(f"推荐计算出错: {e}")
        return jsonify([])

    print(f"耗时: {time.time() - start_t:.2f}s")
    # 直接返回包含原始 URL 的数据
    return jsonify(res)


def recommend_explore_vid():
    global recommender, cookie_str

    if recommender is None:
        print("初始化推荐模型...")
        try:
            recommender = Recommender(cookie_str)
        except Exception as e:
            print(f"模型初始化失败: {e}")
            return jsonify([])
    else:
        recommender.cookies = cookie_str

    print("计算探索推荐...")
    try:
        res = recommender.recommend("recommend", 8)
    except Exception as e:
        print(f"推荐计算出错: {e}")
        return jsonify([])

    return jsonify(res)