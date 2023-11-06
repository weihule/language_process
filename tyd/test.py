import cv2

def main():
    image = cv2.imread('./data/mapofChina.jpg', cv2.IMREAD_UNCHANGED)
    # 获取透明度通道
    alpha_channel = image[:, :, 3]

    # 查看透明度通道，将透明区域显示为黑色，不透明区域显示为白色
    cv2.imshow("Alpha Channel", alpha_channel)

    # 等待用户按下任意键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()