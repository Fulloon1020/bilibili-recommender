import requests
import json


def get_recommand_data(cookie, length):
    url = "https://api.bilibili.com/x/web-interface/wbi/index/top/feed/rcmd"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": cookie,
    }

    res = []
    count = 0
    max_retries = 5
    retries = 0

    while count < length and retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code != 200:
                retries += 1
                continue
            videos_info = response.json()
            if videos_info["code"] != 0:
                retries += 1
                continue
        except:
            retries += 1
            continue

        item_list = videos_info.get("data", {}).get("item", [])
        if not item_list:
            retries += 1
            continue

        for video_info in item_list:
            if count >= length: break

            sigle_res = {}
            sigle_res["bvid"] = video_info["bvid"]
            sigle_res["title"] = video_info["title"]
            sigle_res["pic"] = video_info["pic"]
            sigle_res["author"] = video_info["owner"]["name"]
            sigle_res["view"] = video_info["stat"]["view"]
            sigle_res["like"] = video_info["stat"]["like"]
            sigle_res["duration"] = video_info["duration"]
            sigle_res["tag"] = []  # 置空 Tag

            res.append(sigle_res)
            count += 1

    with open("recvideo.json", "w", encoding="utf-8") as json_file:
        json.dump(res, json_file, indent=4, ensure_ascii=False)

    return res