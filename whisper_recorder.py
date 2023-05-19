import whisper
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import pyaudio
import wave
import audioop
import datetime
import threading
import psutil
import shutil
import traceback
import datetime
import os

# UI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

matplotlib.use('TkAgg')


class AudioRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Recorder")
        self.root.geometry("1400x800")
        self.root.bind('<Button-1>', self.hide_audio_list_popup)
        self.root.bind('<Escape>', lambda event: self.root.quit())
        self.root.bind('<space>', self.toggle_recording)
        
        # Styling
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 16), foreground='black')
        style.configure('TScale', font=('Arial', 16), foreground='black')
        style.configure('TMenubutton', font=('Arial', 16), foreground='black')

       
        self.filename = None
        self.transcript_file = 'transcript.txt'
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SILENCE_THRESHOLD = 5000
        self.SILENCE_DURATION_THRESHOLD = 5  # Seconds of silence before auto-stopping
        self.recording = False
        self.frames = []
        self.record_id = 0
        self.model = 'tiny'

        # Frame for the status bar
        status_frame = tk.Frame(root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Status bar at the left side of the frame
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(status_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X)

        # Popup menu
        self.popup = tk.Menu(self.root, tearoff=0)
        self.popup.add_command(label="Transcribe", command=lambda: self.transcribe_selected_audio())
        self.popup.add_command(label="Delete", command=lambda: self.delete_selected_audio())
        self.popup.add_command(label="Rename", command=lambda: self.rename_selected_audio())
        self.popup.bind("<FocusOut>", self.hide_audio_list_popup)
        
        # Menubar
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Upload Audio", command=self.upload_wav)
        filemenu.add_command(label="Exit", command=self.root.quit)
        devicemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Devices", menu=devicemenu)
        root.config(menu=menubar)

        # Get available devices
        self.available_devices = self.get_available_devices()

        # Add each device to the menu
        default_device = self.available_devices[0] if self.available_devices else None
        for device in self.available_devices:
            devicemenu.add_command(label=device, command=lambda d=device: self.select_device(d))
        if default_device:
            self.select_device(default_device) 

        ## RAM Stuff 
        # RAM info at the right side of the frame
        self.ram_info_var = tk.StringVar()
        self.ram_info_bar = tk.Label(status_frame, textvariable=self.ram_info_var, bd=1, relief=tk.SUNKEN, anchor=tk.E)
        self.ram_info_bar.pack(side=tk.RIGHT, fill=tk.X)

        mem_info = psutil.virtual_memory()
        self.total = mem_info.total / (1024 ** 3)  
        self.available = mem_info.available / (1024 ** 3) 
        self.ram_info_var.set(f"RAM: {round(self.available, 1)} / {round(self.total, 1)}")

        ##  Model menu
        modelmenu = tk.Menu(menubar, tearoff=0)

        # Model options
        self.models = [
            {"name": "tiny", "ram": 1},
            {"name": "base", "ram": 1},
            {"name": "small", "ram": 2},
            {"name": "medium", "ram": 5},
            {"name": "large", "ram": 10},
        ]

        # Add each model to the menu
        default_model = self.models[0]  # Select 'tiny' model by default
        for model in self.models:
            state = "normal" if model["ram"] <= self.available/2 else "disabled"
            modelmenu.add_command(label=f"{model['name']} ({model['ram']}GB of VRAM required)", command=lambda m=model: self.select_model(m), state=state)
        self.select_model(default_model)  # Select the default model

        menubar.add_cascade(label="Model", menu=modelmenu)


        # Frame for buttons
        button_frame = ttk.Frame(root)
        button_frame.pack(side='top', fill='x', padx=5, pady=5)

        # Recording indicator
        self.recording_indicator = tk.Canvas(button_frame, width=30, height=30, bd=0, highlightthickness=0)
        self.recording_dot = self.recording_indicator.create_oval(5, 5, 25, 25, fill='gray')
        self.recording_indicator.pack(side='left', padx=5)

        # Record button
        self.record_button = ttk.Button(button_frame, text='record', command=self.toggle_recording)
        self.record_button.pack(side='left', padx=5)

        # Upload button
        self.upload_button = ttk.Button(button_frame, text='upload', command=self.upload_wav)
        self.upload_button.pack(side='right', padx=5)

        ## Silence threshold slider
        # Frame to hold slider_frame
        grid_frame = tk.Frame(root)
        grid_frame.pack(side='top', fill='x', padx=5, pady=5)

        # Slider frame
        slider_frame = ttk.Frame(grid_frame)
        slider_frame.grid(sticky='ew')

        # Make the column expand
        grid_frame.grid_columnconfigure(0, weight=1)

        self.silence_slider = ttk.Scale(slider_frame, from_=1, to=5, orient='horizontal', command=self.update_silence_duration)
        self.silence_slider.grid(row=0, column=0, sticky='ew')

        # Make the slider's column expand
        slider_frame.grid_columnconfigure(0, weight=1)

        self.silence_slider.set(self.SILENCE_DURATION_THRESHOLD)

        self.silence_label = ttk.Label(slider_frame, text="Silence Duration (s): 5")
        self.silence_label.grid(row=0, column=1, padx=5)

        # Main area frame
        main_frame = tk.Frame(root)
        main_frame.pack(side='top', fill='both', expand=True)

        # Audio list frame
        audio_list_frame = tk.Frame(main_frame)
        audio_list_frame.pack(side='left', fill='both', expand=True)

        # Audio list
        self.audio_list = tk.Listbox(audio_list_frame, font=("Arial", 16), fg="white", selectbackground="green")
        self.audio_list.configure(background="#3d3d3d")
        self.audio_list.bind("<Button-3>", self.audio_list_popup)
        self.audio_list.bind("<Double-Button-1>", self.rename_selected_audio)
        self.audio_list.bind("<Delete>", self.delete_selected_audio) 
        self.audio_list.pack(side='left', fill='both', expand=True, padx=0, pady=0)
        self.update_audio_list()

        # Sound Wave Viz
        self.figure, self.ax = plt.subplots()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.graph = FigureCanvasTkAgg(self.figure, main_frame)
        self.line, = self.ax.plot(np.random.rand(self.CHUNK), color='white', linewidth=3)
        self.ax.set_facecolor('#3d3d3d')
        self.ax.set_ylim([-2 ** 12, (2 ** 12) - 1])

        # Remove ticks from the X and Y axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Repack the graph to make sure it fills all available space
        self.graph.get_tk_widget().pack(side='left', fill='both', expand=True)

        # Transcribe button
        self.transcribe_button = ttk.Button(root, text='Transcribe', command=self.transcribe_selected_audio)
        self.transcribe_button.pack(side='bottom', fill='x', padx=0, pady=0)
        


    def get_available_devices(self):
        p = pyaudio.PyAudio()
        devices = []
        device_count = p.get_device_count()
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            devices.append(device_info["name"])
        p.terminate()
        return devices

    def log_error(self, error):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"[{timestamp}] {str(error)}"
        with open("log.txt", "a") as file:
            file.write(error_message + "\n")

    def select_device(self, device):
        self.update_status(f"Device switched to {device}")

    def select_model(self, model):
        self.model = model
        self.update_status(f"Model switched to {model['name']}")


    def update_status(self, status):
        self.status_var.set(status)

    def reset_plot(self):
        straight_line = np.zeros(self.CHUNK)
        self.line.set_ydata(0)
        self.figure.canvas.draw()

    def plot_data(self, data):
        data_int = np.frombuffer(data, dtype=np.int16)
        # Use the Savitzky-Golay filter to smooth the data
        if len(data_int) >= 10:
            data_int_smooth = savgol_filter(data_int, window_length=35, polyorder=2)
        else:
            data_int_smooth = data_int
        if data_int_smooth.shape[0] < self.CHUNK:
            data_int_smooth = np.pad(data_int_smooth, (0, self.CHUNK - data_int_smooth.shape[0]))
        elif data_int_smooth.shape[0] > self.CHUNK:
            data_int_smooth = data_int_smooth[:self.CHUNK]
        self.line.set_ydata(data_int_smooth)
        self.figure.canvas.draw()

    def audio_list_popup(self, event):
        try:
            self.popup.tk_popup(event.x_root, event.y_root)
        finally:
            self.popup.grab_release()

    def hide_audio_list_popup(self, event=None):
        # Unpost the context menu
        if self.popup is not None:
            self.popup.unpost()

    def transcribe_selected_audio(self):
        selected_audio = self.audio_list.get(self.audio_list.curselection())
        if selected_audio:
            self.filename = selected_audio
            self.start_transcribing()
            self.update_status("Finished Transcribing")
        else:
            self.update_status("No audio file selected.")

    def delete_selected_audio(self, event=None):
        selected_audio = self.audio_list.get(self.audio_list.curselection())
        if selected_audio:
            os.remove(selected_audio)
            self.update_audio_list()
    
    def rename_selected_audio(self, event=None):
        selected_audio = self.audio_list.get(self.audio_list.curselection())
        if selected_audio:
            new_name = simpledialog.askstring("Rename", "Enter new name:", initialvalue=selected_audio)
            if new_name:
                # Append '.wav' extension if not already present
                if not new_name.endswith('.wav'):
                    new_name += '.wav'
                os.rename(selected_audio, new_name)
                self.update_audio_list()


    def toggle_recording(self):
        if self.recording:
            self.recording_indicator.itemconfig(self.recording_dot, fill='white')
            self.recording = False
            self.record_button.config(text='Record')
            self.line.set_color('gray')
            self.reset_plot()

            response = messagebox.askquestion("Save Recording", "Do you want to save the recorded audio?")
            if response == "yes":
                self.save_audio()
                self.update_audio_list()
        else:
            self.filename = f'audio_{str(self.record_id).zfill(3)}.wav'
            self.record_id += 1
            self.recording_indicator.itemconfig(self.recording_dot, fill='green')
            self.recording = True
            self.record_button.config(text='Stop')
            self.line.set_color('green')
            threading.Thread(target=self.record_audio).start()

    def record_audio(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        silent_frames = 0

        while self.recording:

            data = stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            self.plot_data(data)

            # Calculate silent duration
            silent_duration = (silent_frames * self.CHUNK) / self.RATE

            if audioop.rms(data, 4) < self.SILENCE_THRESHOLD:
                silent_frames += 1

            if audioop.rms(data, 2) > self.SILENCE_THRESHOLD:
                silent_frames = 0

            if silent_duration > self.SILENCE_DURATION_THRESHOLD:
                self.recording = False
                self.toggle_recording() 

        stream.stop_stream()
        stream.close()
        p.terminate()
            

    def save_audio(self):
        p = pyaudio.PyAudio()
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.frames = []
        self.update_status(f"Saved audio to {os.path.abspath(self.filename)}")


    def update_silence_duration(self, value):
        self.SILENCE_DURATION_THRESHOLD = int(float(value))
        self.silence_label.config(text=f"Silence Duration (s): {int(float(value))}")


    def upload_wav(self):
        filename = filedialog.askopenfilename(filetypes=[('WAV Files', '*.wav')])
        if filename:
            basename = os.path.basename(filename)
            new_filename = os.path.join(".", basename)
            shutil.copyfile(filename, new_filename)
            self.filename = new_filename
            self.update_status(f"Uploaded {self.filename}")
            self.update_audio_list()


    def update_audio_list(self):
        self.audio_list.delete(0, 'end')
        for file in os.listdir("."):
            if file.endswith(".wav"):
                self.audio_list.insert('end', file)

    def start_transcribing(self):
        if self.filename:
            threading.Thread(target=self.transcribe_audio).start()

    def transcribe_audio(self):
        self.update_status("Transcribing...")
        try:
            model = whisper.load_model(self.model['name'])
        except Exception as e:
            error_message = f"Error loading model: {e}"
            self.update_status(f"Error loading model: read log.txt at root")
            self.log_error(error_message)
            return
        result = model.transcribe(self.filename)
        self.save_transcript(result)

    def save_transcript(self, transcript):
        audio_name, audio_ext = os.path.splitext(self.filename)
        transcript_filename = f"{audio_name}.txt"

        with open(transcript_filename, "w") as file:
            file.write(transcript["text"])
        self.update_status(f"Transcript saved as {transcript_filename} in the current directory.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()