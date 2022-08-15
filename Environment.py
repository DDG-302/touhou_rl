from ctypes import windll, c_uint, c_int, byref, sizeof
from time import sleep
import win32gui
import win32ui
import win32con
import psutil
import pickle as pkl
import Move
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
        self.life_count = self.__get_life()
        self.done = False

        # todo: 还需插入重开指令
        Move.click_with_scane_code(0x48)
        Move.click_with_scane_code(0x2c)

        sleep(2) # 等待重开

        Move.PressKey(0x2c)

        return config.alive_reward, self.__get_img(), False

        

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

        # 释放内存
        # ref: https://www.cnblogs.com/HuaNeedsPills/p/10329763.html
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) # (HWC)
        img = cv2.resize(img, (config.game_scene_resize_to[0], config.game_scene_resize_to[1]))
        img = np.expand_dims(img, 0)
        
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hWnd, hWndDC)
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
        return: (reward, next_img, is_dead)
        - next_img: this is not the state, 
                    you should feed this next_img to the POLICY,
                    the POLICY will handle the state
        - is_dead: is *last* step cause dead or not
        '''
        win_handle = win32gui.FindWindow(0, u"搶曽抧楈揳丂乣 Subterranean Animism. ver 1.00a")
        if(win_handle == 0):
            raise Exception("game crashed...")
        current_life_count = self.__get_life()    
        is_dead = False
        if(current_life_count < self.life_count):
            is_dead = True
        Move.move([action+1])         
        if(current_life_count < 0):
            if(current_life_count < 0):
                self.done = True
                Move.ReleaseKey(0x2C)
            reward = config.dead_penalty
            self.life_count = current_life_count  
        else:
            reward = config.alive_reward
        return reward, self.__get_img(), is_dead

