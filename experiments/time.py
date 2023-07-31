import time

# 获取当前的Unix时间戳（以秒为单位）
timestamp = time.time()

# 如果你需要获取当前的Unix时间戳（以毫秒为单位）
timestamp_milliseconds = int(timestamp * 1000)

# 如果你需要获取当前的Unix时间戳（以微秒为单位）
timestamp_microseconds = int(timestamp * 10**6)

# 如果你需要获取当前的Unix时间戳（以纳秒为单位）
timestamp_nanoseconds = int(timestamp * 10**9)

print(f"Current timestamp in seconds: {timestamp}")
# print(f"Current timestamp in milliseconds: {timestamp_milliseconds}")
# print(f"Current timestamp in microseconds: {timestamp_microseconds}")
# print(f"Current timestamp in nanoseconds: {timestamp_nanoseconds}")
# timestamp1 = 1689560821953502720

# timestamp2 = 1689468026254235478

# difference_in_nanoseconds = timestamp2 - timestamp1
# difference_in_seconds = difference_in_nanoseconds / (10 ** 9)
# difference_in_hours = difference_in_seconds / 3600

# print(f"The difference between the two timestamps is {difference_in_hours} hours.")
