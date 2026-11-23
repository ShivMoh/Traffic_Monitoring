import datetime

def get_current_time():
    time = str(datetime.datetime.now().time())
    return time

def str2time(str : str) -> datetime.datetime:
    return datetime.datetime.strptime(str, "%H:%M:%S.%f")

def time2sec(time : datetime.datetime) -> float:
    return time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1_000_000
