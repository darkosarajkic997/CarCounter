from CarCounter import CarCounter

if __name__ == '__main__':

    VIDEO_ADR = 'AlibiShort.mp4'
    DETECTION_ZONE = (100, 400, 680, 500)
    cc = CarCounter(video_adr=VIDEO_ADR, detection_zone=DETECTION_ZONE)

    cc.count_cars()
