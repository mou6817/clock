import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.umath import rint

for clock in range(50):
    img = cv2.imread("C:/Users/user/Desktop/untitled/clock/"+ str(clock)+'.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # 直接假設圓心
    h, w = img_gray.shape
    x, y = w // 2, h // 2
    r = w // 2

    # 計算平均色彩值
    mean_value = img_bw.mean()

    # 判斷底色(有些錶盤是黑色的)
    if mean_value > 50: 
        inverted_image = img_gray

    else:
        # 如果底盤是黑色的，將顏色反轉
        inverted_image = cv2.bitwise_not(img_bw)
    # 抓黑色區域
    mask = cv2.inRange(inverted_image, 0, 80)

    # 保留中心區域
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    inner_mask = (dist < 85).astype(np.uint8)*255
    mask = cv2.bitwise_and(mask, inner_mask)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    keep = np.zeros_like(mask)

    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8) * 255

        # 這個 component 到中心的最近距離
        ys, xs = np.where(component > 0)
        min_dist = np.min(np.sqrt((xs - x)**2 + (ys - y)**2))

    # 只保留碰到中心附近的區域，也就是指針
        if min_dist < 15:
            keep[labels == i] = 255

    mask = keep
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lines = cv2.HoughLinesP(mask , rho=1, theta=np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)
        # 找時鐘上的線
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] #假設lines.shape=(3,1,4), line[0]就會依序把(1,1,4) (2,1,4) (3,1,4)裡的4個數字取出來，分別對應到x1,y1,x2,y2
            cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 3)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    else:
        print(f"{clock}.png 沒有抓到線，跳過")
        continue

    # 計算線的長度
    line_info = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_info.append([x1, y1, x2, y2, length])

    # 計算線是否穿過圓心or距離圓心很近 (排除其他不是針的線)
    def distance_to_center(x1,x2,y1,y2,cx,cy):

        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        distance = abs(A*cx + B*cy + C) / np.sqrt(A*A + B*B)
        return distance

    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = distance_to_center(x1, x2, y1, y2, x, y)
        # print(f"length={length:.1f}, distance={distance:.1f}")
        
    # 計算每條線的角度
    # 透過計算線條的角度來去判斷哪些線是在同一個針上
    line_features = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        d1 = np.sqrt((x1 - x)**2 + (y1 - y)**2)
        d2 = np.sqrt((x2 - x)**2 + (y2 - y)**2)
        if d1 < d2: # 看哪個端點離圓心較遠就視為針尖
            tip_x, tip_y = x2, y2
        else:        
            tip_x, tip_y = x1, y1

        angle = np.degrees(np.arctan2(tip_y - y, tip_x - x)) % 360
        angle = (angle + 90) % 360 # 把0度轉到12點鐘方向
        line_features.append([tip_x, tip_y, length, angle])
    def angle_diff(a, b):
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def circular_mean(angles):
        angles = np.deg2rad(angles)
        x = np.mean(np.cos(angles))
        y = np.mean(np.sin(angles))
        mean_angle = np.rad2deg(np.arctan2(y, x))
        return mean_angle

    

    # 根據角度將線分成三組，分別對應時針、分針、秒針
    line_features.sort(key = lambda x:x[2], reverse = True)
    hand_groups = []

    for feature in line_features:
        tip_x, tip_y, length, angle = feature
        added = False
        for group in hand_groups:
            group_angle = circular_mean([g[3] for g in group])
            if angle_diff(angle, group_angle) < 8: # 如果角度差距小於15度，就認為是同一個針
                group.append(feature)
                added = True
                break

        if not added:
            hand_groups.append([feature])
    # 取平均
    final_hands = []
    # 避免359度和1度平均變180度

    for group in hand_groups:
        max_length = max([feature[2] for feature in group])
        avg_angle = circular_mean([feature[3] for feature in group])
        avg_angle = (avg_angle+360) % 360 # 將角度限制在0~360度之間
        if max_length < r * 0.15: # 濾掉雜訊
            continue
        final_hands.append((max_length, avg_angle))
    final_hands.sort(key=lambda x: x[0], reverse=True)
    final_hands = final_hands[:2]
    
    use_special = False
    if len(final_hands) == 2:
        len1, angle1 = final_hands[0]
        len2, angle2 = final_hands[1]
        diff = angle_diff(angle1, angle2)
        length_ratio = len2 / len1

        # 只有兩支線長度很接近，而且角度幾乎反方向，才用候選法
        if length_ratio > 0.85 and abs(diff - 180) < 5:# 幾乎反方向，代表抓到同一條線兩端
            use_special = True
            a = final_hands[0][1]
            b = final_hands[1][1]

            # 其中一邊會是分針，另一邊會是時針
            # 但長度排序分不出來，所以要用「合理性」選
            candidates = []
            if use_special:
                for minute_angle, hour_angle in [(a, b), (b, a)]:
                    minute = round(minute_angle / 6)
                    minute = round(minute / 5)*5 # 四捨五入到最近的5分鐘
                    minute = minute % 60 # 控制在 0~59
                    hour = round((hour_angle - minute * 0.5) / 30) % 12 # 把分鐘造成的時針偏移扣掉 並控制在0~11

                    expected_hour_angle = (hour * 30 + minute * 0.5) % 360 # 去計算以目前預設的時針分針，時針角度會是多少
                    err = angle_diff(hour_angle, expected_hour_angle)

                    candidates.append((err, minute_angle, hour_angle, hour, minute))

                best = min(candidates, key=lambda t: t[0]) # 選誤差最小值
                _, minute_angle, hour_angle, hour, minute = best

                if hour == 0:
                    hour = 12

                print(f"{hour}:{minute:02d}")
                continue

    if len(final_hands) == 1: # 若只抓到一支針，假設反方向是另一支針
        angle = final_hands[0][1]
        final_hands.append((final_hands[0][0] * 0.7, (angle + 180) % 360)) # 長度*0.7 是因為時真會比分針短 塞給他的數字， 角度 +180 => 反方向的角 % 360 => 讓角度維持在 0~ 359
        # print(final_hands)
   
    elif len(final_hands) < 2:
        print(f"{clock}.png 只抓到 {len(final_hands)} 支針，跳過")
        continue

    # 角度轉時間
    minute_angle = final_hands[0][1]
    hour_angle = final_hands[1][1]

    minute = round(minute_angle / 6)
    minute = round(minute / 5)*5 # 四捨五入到最近的5分鐘
    minute = minute % 60 # 確保分鐘在0-59之間
    
    hour_len = final_hands[1][0]

    # 修正某些時鐘的時針被判成反方向的情況
    if hour_len < 65 and angle_diff(hour_angle, (minute_angle + 180) % 360) < 8:
        hour_angle = (hour_angle + 180) % 360

    hour = round((hour_angle - minute * 0.5) / 30) % 12
    if hour == 0:
        hour = 12

    # 修 12:10 被判成 11:10 這種情況
    if hour == 11 and minute <= 15 and hour_angle > 345:
        hour = 12
    
    print(f"{hour}:{minute:02d}")
    # print("true:1:05")
    # print("pred:", f"{hour}:{minute:02d}")
    # print("final_hands:", final_hands)
    # print("minute_angle:", minute_angle)
    # print("hour_angle:", hour_angle)
