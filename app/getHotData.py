import requests
import json


def get_hot_data(cookie, length):
    # 改用 B 站综合热门接口，字段包含 'pic'，不会报错
    url = "https://api.bilibili.com/x/web-interface/popular"
    params = {
        "ps": 50,  # 获取数量
        "pn": 1  # 页码
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": cookie,
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code != 200: return []
        data_json = response.json()
        if data_json["code"] != 0: return []
    except:
        return []

    if "list" not in data_json["data"] or not data_json["data"]["list"]:
        return []

    res = []
    count = 0

    for video_info in data_json["data"]["list"]:
        if count >= length: break

        try:
            sigle_res = {}
            sigle_res["bvid"] = video_info["bvid"]
            sigle_res["title"] = video_info["title"]
            # 这里的接口返回的就是 pic，与代码兼容
            sigle_res["pic"] = video_info["pic"]
            sigle_res["author"] = video_info["owner"]["name"]
            sigle_res["view"] = video_info["stat"]["view"]
            sigle_res["like"] = video_info["stat"]["like"]
            sigle_res["favorite"] = video_info["stat"]["favorite"]
            sigle_res["coin"] = video_info["stat"]["coin"]
            sigle_res["share"] = video_info["stat"]["share"]
            sigle_res["duration"] = video_info["duration"]
            sigle_res["pubdate"] = video_info.get("pubdate", 0)

            # 热门接口不返回 Tag，这里置空，由 NLP 模型处理
            sigle_res["tag"] = []

            res.append(sigle_res)
            count += 1
        except Exception as e:
            print(f"解析热门视频出错: {e}")
            continue

    # 缓存
    with open("hotVideo.json", "w", encoding="utf-8") as json_file:
        json.dump(res, json_file, indent=4, ensure_ascii=False)

    return res