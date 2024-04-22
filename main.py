import tkinter as tk
from tkinter import ttk, filedialog
import io
from resize_app_helper import *
from ultralytics import YOLO
from datetime import datetime, timedelta

class ResizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resizing App")
        self.folder_path = "select folder path using the button on the left"
        self.reso_dict = {}
        self.detect_dress = 1
        
        # Top Toolbar
        self.top_frame = tk.Frame(root, height=100)
        self.top_frame.pack(side=tk.TOP)
        self.top_frame.columnconfigure(1, weight=4, minsize=300)
        
        self.filedir_button = tk.Button(self.top_frame, text="Folder..", command=self.select_folder)
        self.filedir_label = tk.Label(self.top_frame, bg='white')
        self.open_button = tk.Button(self.top_frame, text='Edit Resolutions', command=self.open_file)
        self.save_button = tk.Button(self.top_frame, text='Save Resolutions', command=self.save_file)
        
        self.filedir_button.grid(row=0, column=0, pady=3)
        self.filedir_label.grid(row=0, column=1,columnspan=3, padx=5, pady=3)
        self.open_button.grid(row=0, column=4, sticky=tk.NE, pady=3)
        self.save_button.grid(row=0, column=5, sticky=tk.NE, pady=3)
        
        self.run_button = tk.Button(self.top_frame, text='Run', command=self.run)
        self.run_button.grid(row=0, column=6)
        
        self.progress_bar = ttk.Progressbar(self.top_frame, length=100)
        self.progress_bar.grid(row=1, column=6)
        #self.progress_label = tk.Label(self.top_frame)
        #self.progress_label.grid(row=1, column=5)
        
        # Bottom Frame
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM)
        self.text_widget = tk.Text(self.bottom_frame, wrap=tk.WORD)
        self.text_widget.grid(row=0)  

        self.start_up()
                
    def start_up(self):
        self.filedir_label.config(text=self.folder_path)
        a = self.read_reso('resolutions.txt')
        self.reso_dict = self.process_reso(a)
        with open('startup.txt', 'r') as file:
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, file.read())
            
    def read_reso(self, filepath):
        a = []
        with open(filepath, 'r') as f:
            count = 0
            for row in f:
                if '---' in row:
                    count += 1
                elif count >= 2:
                    a.append(str(row).rstrip('\n'))
        return a
        
    def process_reso(self, a):
        d = {}
        curr_key = ''
        for v in a:
            print(v)
            v_list = v.split(',')
            print(v_list)
            if not v_list[0].isnumeric():
                d[v_list[0]] = []
                curr_key = v_list[0]
            else:
                d[curr_key].append((int(v_list[1]), int(v_list[0])))
        return d
    
    def open_file(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "r") as file:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert(tk.END, file.read())
    
    def save_file(self):
        file_path = tk.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(self.text_widget.get("1.0", tk.END))
        reso = self.read_reso(file_path)
        self.reso_dict = self.process_reso(reso)
        self.start_up()
    
    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        self.filedir_label.config(text=self.folder_path)
    
    def run(self):
        model = YOLO('models/best.pt')
        person_model = YOLO('models/yolov8l.pt')
        top_bottom_classes = list(range(13))
        #if not self.detect_dress:
        #    top_bottom_classes = list(range(9))
        file_paths = get_filepaths(self.folder_path + '/')
        folder_name = process_images_path(file_paths)
        start_time = datetime.now()
        print(f'Starting Time: {str(start_time)}')
        for i in range(len(file_paths)):
            s = f' =============== Progress : {i+1}/{len(file_paths)} files completed ==============='
            file = file_paths[i]
            process_image(file, self.reso_dict, folder_name)
            print(s)
            # try:
            #     process_image(file, self.reso_dict, folder_name)
            # except Exception as e:
            #     s = "Error {0}".format(str(e))
            
            #self.write_progress(s)
        print(f" ***************** {i+1} / {i+1} files completed ***************** ")
        end_time = datetime.now()
        print(f'Ending Time: {str(end_time)}')
        time_delta = str(timedelta(seconds=(end_time-start_time).total_seconds()))
        print(f'Time Taken : {str(time_delta)}')
    
    def write_progress(self, s):
        self.text_widget.insert(tk.END, s)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ResizeApp(root)
    root.mainloop()