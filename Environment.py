from ctypes import windll, c_uint, c_int, byref, sizeof
from time import sleep
import win32gui
import win32ui
import win32con
import psutil
import pickle as pkl
import move
import cv2
import numpy as np
import config
# 残机内存地址：4871960

PROCESS_VM_READ = 0x0010 # required to read memory in a process using ReadProcessMemory
LIFE_ADDR = 4871960

win_handle = win32gui.FindWindow(0, u"搶曽抧楈揳丂乣 Subterranean Animism. ver 1.00a")

class TouhouEnvironment:
    def __init__(self) -> None:
        pid = None
        for proc in psutil.process_iter():
            if proc.name() == "th11p1.03.exe":
                pid = proc.pid
        if(pid is None):
            raise Exception("progress not found, you should run the game first")
        
        self.hProcess = windll.kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
        self.life_count = 0
        self.done = False

    def reset(self):
        '''
        return: reward, init_img
        '''
        self.life_count =self.__get_life()
        self.done = False

        # todo: 还需插入重开指令

        sleep(2) # 等待重开

        return config.alive_reward, self.__get_img()

        

    def __get_img(self):
        # ref:https://www.jb51.net/article/199495.htm

        hWnd = win32gui.FindWindow(0, u"搶曽抧楈揳丂乣 Subterranean Animism. ver 1.00a")
        # rec = win32gui.GetWindowRect(win_handle)

        width = 580
        height = 670

        #返回句柄窗口的设备环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
        hWndDC = win32gui.GetWindowDC(hWnd)
        #创建设备描述表
        mfcDC = win32ui.CreateDCFromHandle(hWndDC)
        #创建内存设备描述表
        saveDC = mfcDC.CreateCompatibleDC()
        #创建位图对象准备保存图片
        saveBitMap = win32ui.CreateBitmap()
        #为bitmap开辟存储空间
        saveBitMap.CreateCompatibleBitmap(mfcDC,width,height)
        #将截图保存到saveBitMap中
        saveDC.SelectObject(saveBitMap)
        #保存bitmap到内存设备描述表
        saveDC.BitBlt((0,0), (width,height), mfcDC, (40, 50), win32con.SRCCOPY)

        signedIntsArray = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype=np.uint8)
        img.shape = (height, width, 4)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) # (HWC)
        img = img.shape = (1, 670, 580)
        return img

    def __get_life(self):
        buffer = c_int()
        lpBuffer = byref(buffer)
        nSize = sizeof(buffer)
        lpNumberOfBytesRead = c_uint(0)
        windll.kernel32.ReadProcessMemory(self.hProcess, LIFE_ADDR, lpBuffer, nSize, lpNumberOfBytesRead)
        return buffer.value

    def step(self, action):
        '''
        action: 0, 1, 2, 3, 4 -> up, down, left, right, idle
        return: (reward, next_img)
        - next_img: this is not the state, 
                    you should feed this next_img to the POLICY,
                    the POLICY will handle the state
        '''
        move.move([action+1])
        current_life_count = self.__get_life()
        if(current_life_count < self.life_count):
            if(current_life_count < 0):
                self.done = True
            reward = config.dead_penalty
            self.life_count = current_life_count
        else:
            reward = config.alive_reward
        return reward, self.__get_img()
