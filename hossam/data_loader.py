import requests
import json
from os.path import join, exists
from io import BytesIO
from pandas import DataFrame, read_csv, read_excel

BASE_URL = "https://data.hossam.kr"


def __get_df(path: str, index_col=None) -> DataFrame:
    p = path.rfind(".")
    exec = path[p+1:].lower()

    if exec == 'xlsx':
        # If path is a remote URL, fetch the file once and reuse the bytes
        if path.lower().startswith(('http://', 'https://')):
            path = path.replace("\\", "/")
            with requests.Session() as session:
                r = session.get(path)

                if r.status_code != 200:
                    raise Exception(f"HTTP {r.status_code} Error - {r.reason} > {path}")

                data_bytes = r.content

            # Use separate BytesIO objects for each read to avoid pointer/stream issues
            df = read_excel(BytesIO(data_bytes), index_col=index_col)

            try:
                info = read_excel(BytesIO(data_bytes), sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                from .util import hs_pretty_table
                hs_pretty_table(info)
                print()
            except Exception:
                print(f"\033[91m[!] Cannot read metadata\033[0m")
        else:
            df = read_excel(path, index_col=index_col)

            try:
                info = read_excel(path, sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                from .util import hs_pretty_table
                hs_pretty_table(info)
                print()
            except:
                print(f"\033[91m[!] Cannot read metadata\033[0m")
    else:
        df = read_csv(path, index_col=index_col)

    return df

def __get_data_url(key: str, local: str = None) -> str:
    global BASE_URL

    path = None

    if not local:
        data_path = join(BASE_URL, "metadata.json").replace("\\", "/")

        with requests.Session() as session:
            r = session.get(data_path)

            if r.status_code != 200:
                raise Exception("[%d Error] %s" % (r.status_code, r.reason))

        my_dict = r.json()
        info = my_dict.get(key.lower())

        if not info:
            raise FileNotFoundError("%s는 존재하지 않는 데이터에 대한 요청입니다." % key)

        path = join(BASE_URL, info['url'])
    else:
        data_path = join(local, "metadata.json")

        if not exists(data_path):
            raise FileNotFoundError("존재하지 않는 데이터에 대한 요청입니다.")

        with open(data_path, "r", encoding="utf-8") as f:
            my_dict = json.loads(f.read())

        info = my_dict.get(key.lower())
        path = join(local, info['url'])

    return path, info.get('desc'), info.get('index')


def load_info(search: str = None, local: str = None):
    global BASE_URL

    path = None

    if not local:
        data_path = join(BASE_URL, "metadata.json").replace("\\", "/")

        with requests.Session() as session:
            r = session.get(data_path)

            if r.status_code != 200:
                raise Exception("[%d Error] %s ::: %s" % (r.status_code, r.reason, data_path))

        my_dict = r.json()
    else:
        data_path = join(local, "metadata.json")

        if not exists(data_path):
            raise FileNotFoundError("존재하지 않는 데이터에 대한 요청입니다.")

        with open(data_path, "r", encoding="utf-8") as f:
            my_dict = json.loads(f.read())

    my_data = []
    for key in my_dict:
        if 'index' in my_dict[key]:
            del my_dict[key]['index']

        my_dict[key]['url'] = "%s/%s" % (BASE_URL, my_dict[key]['url'])
        my_dict[key]['name'] = key

        if 'chapter' in my_dict[key]:
            my_dict[key]['chapter'] = ", ".join(my_dict[key]['chapter'])
        else:
            my_dict[key]['chapter'] = '공통'

        my_data.append(my_dict[key])

    my_df = DataFrame(my_data)
    my_df2 = my_df.reindex(columns=['name', 'chapter', 'desc', 'url'])

    if search:
        my_df2 = my_df2[my_df2['name'].str.contains(search.lower())]

    return my_df2


def load_data(key: str, local: str = None):
    index = None
    try:
        url, desc, index = __get_data_url(key, local=local)
    except Exception as e:
        try:
            print(f"\033[91m{str(e)}\033[0m")
        except Exception:
            print(e)
        return

    print("\033[94m[data]\033[0m", url.replace("\\", "/"))
    print("\033[94m[desc]\033[0m", desc)

    df = None

    try:
        df = __get_df(url, index_col=index)
    except Exception as e:
        try:
            print(f"\033[91m{str(e)}\033[0m")
        except Exception:
            print(e)
        return


    return df