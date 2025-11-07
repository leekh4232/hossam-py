import requests
import json
from os.path import join
from tabulate import tabulate
from io import BytesIO
from pandas import DataFrame, read_csv, read_excel

BASE_URL = "https://data.hossam.kr"

def my_pretty_table(data: DataFrame, headers: str = "keys") -> None:
    tabulate.WIDE_CHARS_MODE = False
    print(
        tabulate(
            data, headers="keys", tablefmt="simple", showindex=True, numalign="right"
        )
    )

def get_df(path: str, index_col=None) -> DataFrame:
    p = path.rfind(".")
    exec = path[p+1:].lower()

    if exec == 'xlsx':
        # If path is a remote URL, fetch the file once and reuse the bytes
        if path.lower().startswith(('http://', 'https://')):
            with requests.Session() as session:
                r = session.get(path)

                if r.status_code != 200:
                    print(f"[error] HTTP {r.status_code} {r.reason}")
                    return None

                data_bytes = r.content

            # Use separate BytesIO objects for each read to avoid pointer/stream issues
            df = read_excel(BytesIO(data_bytes), index_col=index_col)

            try:
                info = read_excel(BytesIO(data_bytes), sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                my_pretty_table(info)
                print()
            except Exception:
                print(f"\033[91m[!] Cannot read metadata\033[0m")
        else:
            df = read_excel(path, index_col=index_col)

            try:
                info = read_excel(path, sheet_name='metadata', index_col=0)
                #print("\033[94m[metadata]\033[0m")
                print()
                my_pretty_table(info)
                print()
            except:
                print(f"\033[91m[!] Cannot read metadata\033[0m")
    else:
        df = read_csv(path, index_col=index_col)

    return df

def get_data_url(key: str, local: str = None) -> str:
    global BASE_URL

    path = None

    if not local:
        data_path = join(BASE_URL, "metadata.json")

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

        with open(data_path, "r", encoding="utf-8") as f:
            my_dict = json.loads(f.read())

        info = my_dict.get(key.lower())
        path = join(local, info['url'])

        if not info:
            raise FileNotFoundError("존재하지 않는 데이터에 대한 요청입니다.")

    return path, info.get('desc'), info.get('index')


def load_data(key: str, local: str = None):
    try:
        url, desc, index = get_data_url(key, local=local)
    except Exception as e:
        try:
            print(f"\033[91m[Error] {str(e)}\033[0m")
        except Exception:
            print(e)
        return

    print("\033[94m[data]\033[0m", url)
    print("\033[94m[desc]\033[0m", desc)

    df = None

    try:
        df = get_df(url)
    except Exception as e:
        try:
            print(f"\033[91m[Error] {str(e)}\033[0m")
        except Exception:
            print(e)
        return


    return df

if __name__ == "__main__":
    load_data("AD_SALES")