import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_session_with_retries(retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_fav_data(user_mid, headers, maxcnt=20):
    """
    获取用户所有收藏夹中的视频信息
    """
    all_videos = []
    session = get_session_with_retries()
    folders_url = "https://api.bilibili.com/x/v3/fav/folder/created/list-all"
    params = {"up_mid": user_mid, "jsonp": "jsonp"}
    try:
        response = session.get(folders_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"获取收藏夹列表失败：{e}")
        return []

    folders_info = response.json()
    if folders_info["code"] != 0:
        print(f"获取收藏夹列表失败，错误信息：{folders_info.get('message', '')}")
        return []

    folders = folders_info["data"]["list"] if "data" in folders_info and folders_info["data"] else []
    print(f"找到 {len(folders)} 个收藏夹")

    for folder in folders:
        folder_id = folder["id"]
        folder_title = folder["title"]
        print(f"正在获取收藏夹 '{folder_title}' (ID: {folder_id}) 的视频")

        page_num = 1
        page_size = 20
        while True:
            resources_url = "https://api.bilibili.com/x/v3/fav/resource/list"
            params = {
                "media_id": folder_id,
                "pn": page_num,
                "ps": page_size,
                "keyword": "",
                "order": "mtime",
                "jsonp": "jsonp",
            }
            try:
                response = session.get(resources_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"获取收藏夹 '{folder_title}' 视频失败：{e}")
                break

            resources_info = response.json()
            if resources_info["code"] != 0:
                break

            resources = resources_info["data"]["medias"] if "data" in resources_info and resources_info["data"] else []
            if not resources:
                break

            for media in resources:
                video = {
                    "bvid": media["bvid"],
                    "title": media["title"],  # 保留标题用于NLP
                    "pic": media["cover"],
                    "author": media["upper"]["name"],
                    "view": media["cnt_info"]["play"],
                    "like": 0,
                    "favorite": 0,
                    "coin": media["cnt_info"]["collect"],
                    "share": 0,
                    "duration": media.get("duration", 0),
                    "progress": 0,
                    "tag": [],
                    "isfaved": 1,
                    "isliked": 0,
                    # [修改点] 增加 view_at，收藏夹中用收藏时间 ctime 代替
                    "view_at": media.get("ctime", 0)
                }

                # 获取详情补全Tag
                detail_info_url = "https://api.bilibili.com/x/web-interface/view/detail?bvid=" + video["bvid"]
                try:
                    resp_detail = session.get(detail_info_url, headers=headers, timeout=10)
                    if resp_detail.status_code == 200:
                        v_data = resp_detail.json()
                        if v_data["code"] == 0:
                            data = v_data["data"]
                            video["duration"] = data["View"]["duration"]
                            video["progress"] = video["duration"]  # 假设收藏即看完
                            if "Tags" in data:
                                video["tag"] = [tag["tag_name"] for tag in data["Tags"]]
                            video["like"] = data["View"]["stat"]["like"]
                            video["favorite"] = data["View"]["stat"]["favorite"]
                            video["share"] = data["View"]["stat"]["share"]
                except:
                    pass

                all_videos.append(video)
                if maxcnt is not None and len(all_videos) >= maxcnt:
                    return all_videos  # 这里的return逻辑可能导致只获取部分，如果想全量建议调整

            if resources_info["data"]["has_more"] == 0:
                break
            page_num += 1

    return all_videos


def get_vote_data(cookie, headers=None):
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Cookie": cookie,
        }
    session = get_session_with_retries()
    try:
        response = session.get("https://api.bilibili.com/x/web-interface/nav", headers=headers, timeout=10)
        user_info = response.json()
        if user_info["code"] == 0:
            user_mid = user_info["data"]["mid"]
            return get_fav_data(user_mid, headers)
    except:
        pass
    return []


def get_history_data(cookie, history_len):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": cookie,
    }

    # 1. 先获取收藏数据
    res = get_vote_data(cookie)
    print(f"已获取{len(res)}条收藏记录")

    # 2. 获取观看历史数据
    count = 0
    page = 1
    # 注意：history_len 是总限制，如果收藏已经很多了，这里可能会被跳过，建议逻辑分开或累加

    while True:  # 改为由内部 count 控制退出
        if len(res) >= history_len:
            break

        url = f"https://api.bilibili.com/x/v2/history?pn={page}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: break
            videos_info = response.json()
            if videos_info["code"] != 0: break
        except:
            break

        data_list = videos_info.get("data", [])
        if not data_list: break

        for video_info in data_list:
            # 过滤非视频内容（如专栏）
            if video_info.get("badge") == "专栏": continue

            sigle_res = {}
            sigle_res["bvid"] = video_info["bvid"]
            sigle_res["title"] = video_info["title"]  # 核心NLP特征
            sigle_res["pic"] = video_info["pic"]
            sigle_res["author"] = video_info["owner"]["name"]
            sigle_res["view"] = video_info["stat"]["view"]
            sigle_res["like"] = video_info["stat"]["like"]
            sigle_res["favorite"] = video_info["stat"]["favorite"]
            sigle_res["coin"] = video_info["stat"]["coin"]
            sigle_res["share"] = video_info["stat"]["share"]
            sigle_res["duration"] = video_info["duration"]
            sigle_res["progress"] = video_info["progress"]
            if sigle_res["progress"] == -1:
                sigle_res["progress"] = sigle_res["duration"]

            # [修改点] 获取观看时间戳，这对序列化推荐至关重要
            sigle_res["view_at"] = video_info.get("view_at", 0)

            # 获取 Tag 详情
            url_2 = "https://api.bilibili.com/x/web-interface/view/detail?bvid=" + video_info["bvid"]
            try:
                resp2 = requests.get(url_2, headers=headers, timeout=5)
                if resp2.status_code == 200:
                    v_detail = resp2.json()
                    if v_detail["code"] == 0:
                        sigle_res["tag"] = [tag["tag_name"] for tag in v_detail["data"].get("Tags", [])]
            except:
                sigle_res["tag"] = []

            sigle_res["isfaved"] = 1 if video_info["favorite"] else 0

            # 简单判断是否点赞（API通常不直接返回历史列表里的点赞状态，需额外请求，为加速此处简化）
            sigle_res["isliked"] = 0

            res.append(sigle_res)
            count += 1
            print(f"已获取历史记录: {count} / {history_len}")

            if len(res) >= history_len:
                break

        page += 1
        if len(res) >= history_len:
            break

    # [修改点]核心步骤：按时间戳从小到大排序（最旧的在前面），用于训练序列模型
    res.sort(key=lambda x: x.get("view_at", 0))

    # 写入文件移到最后，避免IO瓶颈
    with open("historyVideo.json", "w", encoding="utf-8") as json_file:
        json.dump(res, json_file, indent=4, ensure_ascii=False)

    return res