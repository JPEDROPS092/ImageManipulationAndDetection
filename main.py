import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import timedelta, datetime
import pytz
from ultralytics import YOLO
# Dependências
# pip install opencv-python pillow numpy customtkinter pytz

class VideoImageProcessor:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Processador de Imagens e Vídeos")
        
        # Variáveis de controle
        self.current_file = None
        self.is_video = False
        self.cap = None
        self.current_frame = None
        self.original_frame = None
        self.zoomed_frame = None
        self.roi_points = []
        self.drawing_roi = False
        self.recording = False
        self.recording_mode = "video"
        self.webcam_save_dir = ""
        self.webcam_record_frames = 0
        self.save_recording = False
        self.webcam_writer = None
        self.video_cutpoints = []
        self.video_speed = 1.0
        self.image_offset = (0, 0)
        self.is_paused = False
        self.is_video_reverse = False
        self.video_current_frame = 0
        self.zoom_rect = (0, 0, 0, 0)

        self.video_filters = []
        
        self.setup_gui()

    def setup_gui(self):
        # Frame principal usando CTkFrame com fundo escuro
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#2D2D2D")  # Cor de fundo escura

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="news")
        
        # Controles superiores
        self.setup_top_controls()
        
        # Área de visualização
        self.canvas = ctk.CTkCanvas(self.main_frame, width=800, height=600, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, columnspan=1, pady=10, padx=10)

        # Área do ROI
        roi_frame = ctk.CTkFrame(self.main_frame, fg_color="#444444", corner_radius=5)
        roi_frame.grid(row=1, column=1, columnspan=1, pady=10, padx=10)

        # Área de visualização do ROI
        self.roi_canvas = ctk.CTkCanvas(roi_frame, width=400, height=300, bg="black", highlightthickness=0)
        self.roi_canvas.grid(row=0, column=0, columnspan=1, pady=10, padx=10)
        
        # Botão para salvar o ROI
        button_save_roi = ctk.CTkButton(roi_frame, text="Salvar ROI", command=self.save_roi, corner_radius=8, 
                                         width=70, height=20, fg_color="#585858", text_color="white", 
                                         font=("Trebuchet MS", 12, "bold"))
        button_save_roi.grid(row=1, column=0, padx=16, pady=8, sticky="w")  # Adiciona um maior espaçamento entre os widgets
        
        # Controles de filtros e operações
        self.setup_filter_controls()
        
        # Controles de vídeo
        self.setup_video_controls()

        # Carregar modelo YOLOv8
        self.model = YOLO("yolov8n.pt")   

        # Bind eventos do mouse
        self.canvas.bind("<Button-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)

        self.root.bind("<Configure>", self.window_resize)

    # Método para configurar os controles superiores da interface
    def setup_top_controls(self):
        # Criando o quadro de controles superiores com fundo escuro
        controls_frame = ctk.CTkFrame(self.main_frame, fg_color="#444444", corner_radius=5)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1, minsize=120)

        controls_frame.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="new")
        controls_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        # Botão para abrir arquivo
        button_open_file = ctk.CTkButton(controls_frame, text="Abrir Arquivo", command=self.open_file, corner_radius=8, 
                                         width=70, height=20, fg_color="#585858", text_color="white", 
                                         font=("Trebuchet MS", 12, "bold"))
        button_open_file.grid(row=0, column=0, padx=16, pady=8, sticky="w")  # Adiciona um maior espaçamento entre os widgets
        
        # ---- Título e Seleção de Modo ----
        label_mode = ctk.CTkLabel(controls_frame, text="Tipo:", text_color="white", font=("Trebuchet MS", 12, "bold"))
        label_mode.grid(row=0, column=1, padx=16, sticky="e")  # Alinha o título à esquerda com espaçamento
        
        # Variável para armazenar o modo selecionado
        self.mode_var = tk.StringVar(value="image")
        
        # RadioButtons para seleção do modo de imagem ou vídeo
        radiobutton_image = ctk.CTkRadioButton(controls_frame, text="Imagem", variable=self.mode_var, value="image", 
                                               font=("Trebuchet MS", 12, "bold"), command=self.mode_changed)
        radiobutton_image.grid(row=0, column=2, sticky="w")
        
        radiobutton_video = ctk.CTkRadioButton(controls_frame, text="Vídeo", variable=self.mode_var, value="video", 
                                               font=("Trebuchet MS", 12, "bold"), command=self.mode_changed)
        radiobutton_video.grid(row=0, column=3, sticky="w")

        radiobutton_webcam = ctk.CTkRadioButton(controls_frame, text="Webcam", variable=self.mode_var, value="webcam",
                                               font=("Trebuchet MS", 12, "bold"), command=self.open_webcam)
        radiobutton_webcam.grid(row=0, column=4, sticky="w") 

        # ---- Seleção para extrair ROI ou aplicar Zoom ----
        label_mode = ctk.CTkLabel(controls_frame, text="Abrir ROI:", text_color="white", font=("Trebuchet MS", 12, "bold"))
        label_mode.grid(row=0, column=4, padx=16, sticky="e")  # Alinha o título à esquerda com espaçamento

        self.roi_zoom_var = tk.StringVar(value="roi")

        # RadioButtons para seleção do modo de imagem ou vídeo
        radiobutton_image = ctk.CTkRadioButton(controls_frame, text="ROI", variable=self.roi_zoom_var, value="roi", 
                                               font=("Trebuchet MS", 12, "bold"))
        radiobutton_image.grid(row=0, column=5, sticky="w")
        
        radiobutton_video = ctk.CTkRadioButton(controls_frame, text="Zoom", variable=self.roi_zoom_var, value="zoom", 
                                               font=("Trebuchet MS", 12, "bold"))
        radiobutton_video.grid(row=0, column=6, sticky="w")
        
        # ---- Modo de Processamento ----
        label_process_mode = ctk.CTkLabel(controls_frame, text="Processamento:", text_color="white", font=("Trebuchet MS", 12, "bold"))
        label_process_mode.grid(row=1, column=1, padx=24, sticky="e")
        
        # Variável para armazenar o modo de processamento selecionado
        self.processing_mode = tk.StringVar(value="independent")
        
        # RadioButtons para escolher o modo de processamento
        radiobutton_independent = ctk.CTkRadioButton(controls_frame, text="Independente", height=10, width=10, 
                                                     variable=self.processing_mode, value="independent", 
                                                     text_color="white", font=("Trebuchet MS", 12, "bold"))
        radiobutton_independent.grid(row=1, column=2, sticky="w")
        
        radiobutton_cascade = ctk.CTkRadioButton(controls_frame, text="Cascata", variable=self.processing_mode, 
                                                 value="cascade", text_color="white", font=("Trebuchet MS", 12, "bold"))
        radiobutton_cascade.grid(row=1, column=3, sticky="w")
    
    # Método para configurar os controles de filtros
    def setup_filter_controls(self):
        # Criando o quadro de controles de filtro com fundo escuro
        filter_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=5)
        filter_frame.grid(row=2, column=0, pady=10, sticky=tk.W, padx=10)
        
        # ---- Título dos Filtros ----
        label_filters = ctk.CTkLabel(filter_frame, text="Filtros e Operações", text_color="white", 
                                     font=("Trebuchet MS", 12, "bold"))
        label_filters.grid(row=0, column=0, columnspan=4, pady=10, padx=10)
        
        # ---- Controles de Filtro ----
        button_Blur = ctk.CTkButton(filter_frame, text="Blur", command=self.apply_blur, corner_radius=8, 
                                    width=70, height=20, fg_color="#585858", text_color="white", 
                                    font=("Trebuchet MS", 12, "bold"))
        button_Blur.grid(row=1, column=0, padx=5, pady=5)
        
        button_Sharpen = ctk.CTkButton(filter_frame, text="Sharpen", command=self.apply_sharpen, corner_radius=8, 
                                       width=70, height=20, fg_color="#585858", text_color="white", 
                                       font=("Trebuchet MS", 12, "bold"))
        button_Sharpen.grid(row=1, column=1, padx=5, pady=5)
        
        button_Emboss = ctk.CTkButton(filter_frame, text="Emboss", command=self.apply_emboss, corner_radius=8, 
                                      width=70, height=20, fg_color="#585858", text_color="white", 
                                      font=("Trebuchet MS", 12, "bold"))
        button_Emboss.grid(row=1, column=2, padx=5, pady=5)
        
        button_Laplacian = ctk.CTkButton(filter_frame, text="Laplacian", command=self.apply_laplacian, corner_radius=8, 
                                         width=70, height=20, fg_color="#585858", text_color="white", 
                                         font=("Trebuchet MS", 12, "bold"))
        button_Laplacian.grid(row=1, column=3, padx=5, pady=5)
        
        button_Canny = ctk.CTkButton(filter_frame, text="Canny", command=self.apply_canny, corner_radius=8, 
                                     width=70, height=20, fg_color="#585858", text_color="white", 
                                     font=("Trebuchet MS", 12, "bold"))
        button_Canny.grid(row=2, column=0, padx=5, pady=5)

        button_binary = ctk.CTkButton(filter_frame, text="Sobel", command=self.apply_sobel, corner_radius=8, 
                                      width=70, height=20, fg_color="#585858", text_color="white", 
                                      font=("Trebuchet MS", 12, "bold"))
        button_binary.grid(row=2, column=1, padx=5, pady=5)
        
        # ---- Controles de Cor ----
        button_grayscale = ctk.CTkButton(filter_frame, text="Cinza", command=self.convert_grayscale, corner_radius=8, 
                                         width=70, height=20, fg_color="#585858", text_color="white", 
                                         font=("Trebuchet MS", 12, "bold"))
        button_grayscale.grid(row=2, column=2, padx=5, pady=5)
        
        button_binary = ctk.CTkButton(filter_frame, text="Binário", command=self.convert_binary, corner_radius=8, 
                                      width=70, height=20, fg_color="#585858", text_color="white", 
                                      font=("Trebuchet MS", 12, "bold"))
        button_binary.grid(row=2, column=3, padx=5, pady=5)
        
        button_color = ctk.CTkButton(filter_frame, text="Colorido", command=self.restore_color, corner_radius=8, 
                                     width=70, height=20, fg_color="#585858", text_color="white", 
                                     font=("Trebuchet MS", 12, "bold"))
        button_color.grid(row=3, column=0, padx=5, pady=5)

        button_color = ctk.CTkButton(filter_frame, text="Detectar Objetos", command=self.detect_objects, corner_radius=8, 
                                     width=70, height=20, fg_color="#585858", text_color="white", 
                                     font=("Trebuchet MS", 12, "bold"))
        button_color.grid(row=3, column=0, padx=5, pady=5)

        
    
    # Método para configurar os controles de vídeo
    def setup_video_controls(self):
        # Criando o quadro de controles de vídeo
        video_frame = ctk.CTkFrame(self.main_frame, fg_color="#333333", corner_radius=10)  
        video_frame.grid(row=2, column=1, pady=10, sticky=tk.W, padx=10)
        
        # ---- Título dos Controles de Vídeo ----
        label_video_controls = ctk.CTkLabel(video_frame, text="Controles de Vídeo", text_color="white", 
                                            font=("Trebuchet MS", 12, "bold"))
        label_video_controls.grid(row=0, column=0, columnspan=6, pady=5, padx=10)

        # Criando o quadro de exibição da velocidade
        speed_frame = ctk.CTkFrame(video_frame, fg_color="#333333", corner_radius=10,
                                   width=70, height=20)  
        speed_frame.grid(row=3, column=2, columnspan=1, pady=5, padx=5)

        # ---- Label da Velocidade ----
        label_speed = ctk.CTkLabel(speed_frame, text="Velocidade", text_color="white",
                                            font=("Trebuchet MS", 12, "bold"))
        label_speed.grid(row=0, column=0, pady=5, padx=5)

        # ---- Valor da velocidade ----
        self.label_speed_value = ctk.CTkLabel(speed_frame, text="1x", text_color="#8888a3", 
                                            font=("Trebuchet MS", 12, "bold"))
        self.label_speed_value.grid(row=0, column=1, pady=5, padx=5)
        
        # ---- Controles de Velocidade ----
        button_speed_up = ctk.CTkButton(video_frame, text="Acelerar", command=self.speed_up, corner_radius=8, 
                                        width=70, height=20, fg_color="#585858", text_color="white", 
                                        font=("Trebuchet MS", 12, "bold"))
        button_speed_up.grid(row=1, column=0, padx=5, pady=5)
        
        button_pause = ctk.CTkButton(video_frame, text="Pausar", command=self.toggle_pause, corner_radius=8, 
                                     width=70, height=20, fg_color="#585858", text_color="white", 
                                     font=("Trebuchet MS", 12, "bold"))
        button_pause.grid(row=1, column=1, padx=5, pady=5)
        
        button_slow_down = ctk.CTkButton(video_frame, text="Desacelerar", command=self.slow_down, corner_radius=8, 
                                         width=70, height=20, fg_color="#585858", text_color="white", 
                                         font=("Trebuchet MS", 12, "bold"))
        button_slow_down.grid(row=1, column=2, padx=5, pady=5)
        
        # ---- Controles de Direção ----
        button_reverse_direction = ctk.CTkButton(video_frame, text="Inverter Direção", command=self.toggle_direction, 
                                                 corner_radius=8, width=70, height=20, fg_color="#585858", 
                                                 text_color="white", font=("Trebuchet MS", 12, "bold"))
        button_reverse_direction.grid(row=2, column=0, padx=5, pady=5)
        
        # ---- Marcação de Ponto de Corte ----
        button_mark_cutpoint = ctk.CTkButton(video_frame, text="Ponto de Corte", command=self.mark_cutpoint, 
                                             corner_radius=8, width=70, height=20, fg_color="#585858", 
                                             text_color="white", font=("Trebuchet MS", 12, "bold"))
        button_mark_cutpoint.grid(row=2, column=1, padx=5, pady=5)
        
        # ---- Salvar Segmentos ----
        button_save_segments = ctk.CTkButton(video_frame, text="Salvar Segmentos", command=self.save_video_segments, 
                                             corner_radius=8, width=70, height=20, fg_color="#585858", 
                                             text_color="white", font=("Trebuchet MS", 12, "bold"))
        button_save_segments.grid(row=2, column=2, padx=5, pady=5)      

        # ---- Gravar Webcam ----
        self.button_record_webcam = ctk.CTkButton(video_frame, text="Iniciar Gravação", command=self.record_webcam, 
                                             corner_radius=8, width=70, height=20, fg_color="#585858", 
                                             text_color="white", font=("Trebuchet MS", 12, "bold"))
        self.button_record_webcam.grid(row=3, column=0, padx=5, pady=5)      
     
    # Função de controle do tamanho da janela
    def window_resize(self, event: tk.Event):
        #if self.root.winfo_height() < 900:
            self.canvas.configure(height=self.root.winfo_height() - 380)
            self.show_frame()              
    
    # Função de controle da troca de modo (vídeo ou imagem)
    def mode_changed(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.original_frame = None
        self.is_paused = False
        self.zoom_rect = (0, 0, 0, 0)
        self.video_cutpoints = []
        self.canvas.delete("all")
    
    # Funções de abertura de arquivo (geral)
    def open_file(self):
        if self.mode_var.get() == "image":
            filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        elif self.mode_var.get() == "video":
            filetypes = [("Video files", "*.mp4 *.avi *.mov")]
        elif self.mode_var.get() == "webcam":
            self.open_webcam()
            return 
        
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.mode_changed()
            self.current_file = filename
            if self.mode_var.get() == "video":
                self.open_video()
            else:
                self.open_image()
    
    # Funções de abertura de imagem
    def open_image(self):
        try:
            self.current_frame = cv2.imread(self.current_file)
            if self.current_frame is not None:
                self.original_frame = self.current_frame.copy()
                self.show_frame()
            else:
                print("Erro: A imagem não foi carregada corretamente.")
                # Trate o erro ou forneça um valor padrão
                self.original_frame = None
            
        except Exception as e:
            print(f"Erro ao tentar carregar ou processar a imagem: {e}")
            self.original_frame = None
    
    # Funções de abertura de vídeo
    def open_video(self):
        self.cap = cv2.VideoCapture(self.current_file)
        self.update_video_frame()

    # Funções de abertura da Webcam
    def open_webcam(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível acessar a webcam.")
            return
        # Iniciar atualização do frame
        self.update_webcam_frame() 

    def detect_objects(self):
        if self.mode_var.get() == 'image':  # Modo imagem
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame.copy()  # Restaurar o frame original
                
                # Realizar detecção de objetos no frame atual
                results = self.model(self.current_frame, conf=0.5)  # Ajustar confiança conforme necessário
                annotated_frame = results[0].plot()  # Anotar o frame com os resultados
                
                self.current_frame = annotated_frame  # Atualizar o frame atual com as anotações
                self.show_frame()  # Exibir o frame anotado

        elif self.mode_var.get() in ['video', 'webcam']:  # Modo vídeo ou webcam
            if self.cap is not None:
                self.video_filters.clear()  # Limpar filtros aplicados ao vídeo
                self.video_filters.append('detect_objects')  # Adicionar a detecção como um filtro

    # Funções de aplicação de filtro em vídeo
    def apply_filters_on_video(self, frame):
        processed_frame = frame.copy()
        for filter in self.video_filters:
            if filter == 'detect_objects':
                # Realizar detecção de objetos (por exemplo, usando YOLO)
                results = self.model(frame, conf=0.5)  # Ajuste a confiança conforme necessário
                annotated_frame = results[0].plot()  # Marcar os objetos detectados
                frame = annotated_frame  # Substituir o frame com as anotações
                
            if filter == 'blur':
                processed_frame = cv2.GaussianBlur(processed_frame, (5,5), 0)
            
            if filter == 'sharpen':
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed_frame =  cv2.filter2D(processed_frame, -1, kernel)
            
            if filter == 'emboss':
                kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
                processed_frame =  cv2.filter2D(processed_frame, -1, kernel)
            
            if filter == 'laplacian':
                processed_frame =  cv2.Laplacian(processed_frame, cv2.CV_64F).astype(np.uint8)
            
            if filter == 'canny':
                processed_frame = cv2.Canny(processed_frame, 100, 200)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

            if filter == 'sobel':
                grad_x = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                processed_frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            
            if filter == 'gray':
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            if filter == 'binary':
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                processed_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


        return processed_frame
    
    def update_video_frame(self):
        if self.cap is not None and not self.is_paused:
            # Controle de frames para vídeo reverso
            if self.is_video_reverse:
                if self.video_current_frame > 0:
                    self.video_current_frame -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_current_frame)
            else:
                if self.video_current_frame < self.cap.get(cv2.CAP_PROP_POS_FRAMES):
                    self.video_current_frame += 1

            # Ler o próximo frame do vídeo
            ret, frame = self.cap.read()
            if ret:
                self.original_frame = frame.copy()

                # Aplicar filtros no frame (incluindo a detecção de objetos)
                self.current_frame = self.apply_filters_on_video(self.original_frame)
                
                # Se o zoom estiver ativado, aplicar o zoom no frame
                if self.is_zoomed():
                    self.current_frame = self.apply_zoom_video(self.current_frame)

                # Exibir o frame com a detecção
                self.show_frame()

            # Continuar chamando a função após o intervalo determinado pela velocidade do vídeo
            self.root.after(int(30 / self.video_speed), self.update_video_frame)

        elif self.is_paused:
            # Se o vídeo estiver pausado, exibir o último frame (ou o frame original)
            if self.original_frame is not None:
                self.current_frame = self.original_frame.copy()
                # Aplicar filtros ao frame
                self.current_frame = self.apply_filters_on_video(self.current_frame)
                
                # Se o zoom estiver ativado, aplicar o zoom no frame
                if self.is_zoomed():
                    self.current_frame = self.apply_zoom_video(self.current_frame)
                
                # Exibir o frame
                self.show_frame()



    # Funções de atualização do quadro exibido na reprodução da Webcam
    def update_webcam_frame(self):
        if self.cap is not None:
            if self.is_video_reverse:
                if self.video_current_frame > 0:
                    self.video_current_frame -= 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_current_frame)
            else:
                if self.video_current_frame < self.cap.get(cv2.CAP_PROP_POS_FRAMES):
                    self.video_current_frame += 1
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.original_frame = frame.copy()
                self.current_frame = self.apply_filters_on_video(frame)
                if self.is_zoomed():
                    self.current_frame = self.apply_zoom_video(self.current_frame)
                else:
                    self.show_frame()
                if self.recording == True:
                    self.save_webcam_record()
                    
            else:
                print("Erro ao capturar o frame da webcam.")

        # Atualizar novamente após 10ms
        self.root.after(10, self.update_webcam_frame)

    def show_frame(self):
        if self.current_frame is not None:
            # Converter de BGR para RGB para exibição no Tkinter
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

            # Redimensionar a imagem mantendo a proporção
            height, width = frame_rgb.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            self.ratio = max(width / canvas_width, height / canvas_height)
            new_width = int(width / self.ratio)
            new_height = int(height / self.ratio)

            self.image_offset = ((canvas_width - new_width) / 2, (canvas_height - new_height) / 2)

            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

            # Converter para PhotoImage (para exibição no Tkinter)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

            # Mostrar na Canvas
            self.canvas.delete("all")  # Limpar a canvas antes de desenhar a nova imagem
            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                    image=self.photo, anchor=tk.CENTER)

            # Manter uma referência da imagem para evitar que ela seja coletada pelo garbage collector
            self.canvas.image = self.photo


    # ----------- Funções de ROI -------------
    def start_roi(self, event):
        self.roi_points = [(event.x, event.y)]
        self.drawing_roi = True
    
    def draw_roi(self, event):
        if self.drawing_roi:
            self.canvas.delete("roi")
            self.canvas.create_rectangle(self.roi_points[0][0], self.roi_points[0][1],
                                      event.x, event.y, outline="red", tags="roi")
    
    def end_roi(self, event):
        if self.drawing_roi:
            self.roi_points.append((event.x, event.y))
            self.drawing_roi = False
            self.process_roi()

    def save_roi(self):
        # Opção de salvar
        if messagebox.askyesno("Salvar ROI", "Deseja salvar a região selecionada?"):
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
            if filename:
                cv2.imwrite(filename, self.roi_image)
    
    def process_roi(self):
        if len(self.roi_points) == 2:
            # Converter coordenadas do canvas para coordenadas da imagem
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]

            print(self.image_offset[0], self.image_offset[1], self.ratio)

            x1 = int((x1 - self.image_offset[0]) * self.ratio)
            x2 = int((x2 - self.image_offset[0]) * self.ratio)
            y1 = int((y1 - self.image_offset[1]) * self.ratio)
            y2 = int((y2 - self.image_offset[1]) * self.ratio)

            x1 = max(0, x1)
            y1 = max(0, y1)

            if self.roi_zoom_var.get() == 'roi':
                # Extrair ROI
                self.roi_image = self.current_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]

                if messagebox.askyesno("Abrir ROI", "Abrir em janela separada?"):
                    try:
                        roi_ = cv2.resize(self.roi_image, (max(1, int(self.roi_image.shape[1] / self.ratio)), 
                                                        max(1, int(self.roi_image.shape[0] / self.ratio))))
                    except Exception as e:
                        print("Erro ao tentar adquirir ROI")
                        return

                    # Mostrar ROI em nova janela
                    cv2.imshow("Region of Interest", roi_)
                else:            
                    frame_rgb = cv2.cvtColor(self.roi_image, cv2.COLOR_BGR2RGB)

                    # Redimensionar mantendo proporção
                    height, width = frame_rgb.shape[:2]
                    canvas_width = self.roi_canvas.winfo_width()
                    canvas_height = self.roi_canvas.winfo_height()
                    
                    self.roi_ratio = max(width/canvas_width, height/canvas_height)
                    new_width = int(width / self.roi_ratio)
                    new_height = int(height / self.roi_ratio)
                    
                    frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

                    self.roi_photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))

                    # Mostrar na canvas
                    self.roi_canvas.delete("all")
                    self.roi_canvas.create_image(canvas_width//2, canvas_height//2, 
                                        image=self.roi_photo, anchor=tk.CENTER)
                    
                
            elif self.roi_zoom_var.get() == 'zoom':
                if messagebox.askyesno("Zoom", "Deseja aplicar Zoom?"):
                    self.zoom_rect = (x1, x2, y1, y2)
                    if self.is_paused:
                            self.update_video_frame()
                    if self.mode_var.get() == "image":
                        self.apply_zoom_image()


                            
    # ------------- Funções de controle de vídeo --------------
    # Velocidade
    def speed_up(self):
        if self.video_speed < 5:
            self.video_speed *= 1.5
        self.label_speed_value.configure(text="{:.1f}".format(self.video_speed) + 'x')
    
    def slow_down(self):
        if self.video_speed > 0.1:
            self.video_speed *= 0.75
        self.label_speed_value.configure(text="{:.1f}".format(self.video_speed) + 'x')

    # Pausa    
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.update_video_frame()
    
    # Direção de reprodução do vídeo
    def toggle_direction(self):
        if self.cap is not None:
            if self.is_video_reverse:
                self.is_video_reverse = False
            else:
                self.is_video_reverse = True
    
    # Marcar ponto de corte
    def mark_cutpoint(self):
        if self.cap is not None:
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if self.mode_var.get() == "video":
                self.video_cutpoints.append(current_time)
                messagebox.showinfo("Ponto Marcado", 
                                f"Tempo marcado: {timedelta(seconds=int(current_time))}")
    
    def record_webcam(self):
        if self.mode_var.get() != "webcam":
            return
        if self.recording == False:
            save_mode = messagebox.askyesno("Modo de Salvamento", 
                                      "Deseja salvar como frames?\n'Sim' para frames, 'Não' para vídeo")
            
            # Criar diretório para salvar
            self.webcam_save_dir = filedialog.askdirectory(title="Selecione pasta para salvar")
            if not self.webcam_save_dir:
                return
            
            if save_mode:
                self.recording_mode = "frames"

            else:
                self.recording_mode = "video"

            # Obter propriedades do vídeo original
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Grava o arquivo do vídeo salvo
            time_now = datetime.now(pytz.timezone("America/Manaus"))
            output_path = os.path.join(self.webcam_save_dir, f"segment_{time_now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.webcam_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            self.button_record_webcam.configure(text="Parar Gravação")
            self.recording = True
            self.webcam_record_frames = 0

        else:
            self.recording = False
            self.button_record_webcam.configure(text="Iniciar Gravação")
            self.webcam_writer.release()
        
    def save_webcam_record(self):
        self.webcam_record_frames += 1
        if self.recording_mode == "video":
            self.webcam_writer.write(self.current_frame)
        elif self.recording_mode == "frames":
            os.makedirs(os.path.join(self.webcam_save_dir, "recorded_frames"), exist_ok=True)
            cv2.imwrite(os.path.join(self.webcam_save_dir, "recorded_frames", f"frame_{self.webcam_record_frames}.jpg"), 
                        self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        else:
            print("Modo de gravação desconhecido")
    
    # Salvar segmentos cortados (video)
    def save_video_segments(self):
        if not self.video_cutpoints:
            messagebox.showwarning("Aviso", "Nenhum ponto de corte marcado!")
            return
        
        # Ordenar pontos de corte
        self.video_cutpoints.sort()
        
        # Adicionar início e fim do vídeo
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        points = [0] + self.video_cutpoints + [total_duration]
        
        # Perguntar modo de salvamento
        save_mode = messagebox.askyesno("Modo de Salvamento", 
                                      "Deseja salvar como frames?\n'Sim' para frames, 'Não' para vídeo")
        
        # Criar diretório para salvar
        save_dir = filedialog.askdirectory(title="Selecione pasta para salvar")
        if not save_dir:
            return
        
        # Processar cada segmento
        for i in range(len(points) - 1):
            start_time = points[i]
            end_time = points[i+1]
            
            # Configurar posição inicial
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            if save_mode:  # Salvar como frames
                frames_dir = os.path.join(save_dir, f"segment_{i+1}_frames")
                os.makedirs(frames_dir, exist_ok=True)
                frame_count = 0
                
                while self.cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                    frame = self.apply_filters_on_video(frame)
                    if self.is_zoomed():
                        frame = self.apply_zoom_video(frame)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_count += 1
            
            else:  # Salvar como vídeo
                # Obter propriedades do vídeo original
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Configurar writer
                output_path = os.path.join(save_dir, f"segment_{i+1}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                while self.cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame = self.apply_filters_on_video(frame)
                    if self.is_zoomed():
                        frame = self.apply_zoom_video(frame)
                    out.write(frame)
                
                out.release()
        
        messagebox.showinfo("Concluído", "Segmentos salvos com sucesso!")
        self.video_cutpoints = []  # Limpar pontos de corte
     
    def apply_zoom_image(self):
        x1 = self.zoom_rect[0]
        x2 = self.zoom_rect[1]
        y1 = self.zoom_rect[2]
        y2 = self.zoom_rect[3]

        # Aplicando Zoom
        if self.zoomed_frame is not None:
            self.current_frame = self.zoomed_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        else:   
            self.current_frame = self.current_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        
        self.zoomed_frame = self.current_frame.copy()
        self.original_frame = self.zoomed_frame.copy()

        # Redimensionar mantendo proporção
        height, width = self.current_frame.shape[:2]
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.ratio = max(width/canvas_width, height/canvas_height)
        new_width = int(width / self.ratio)
        new_height = int(height / self.ratio)

        self.image_offset = ((canvas_width - new_width) / 2, (canvas_height - new_height) / 2)
        
        frame_resized = cv2.resize(self.current_frame, (new_width, new_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

        # Mostrar na canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                            image=self.photo, anchor=tk.CENTER)
    
    # Aplicar zoom e abrir em nova janela (Retorna frame com zoom)
    def apply_zoom_video(self, original_frame):
        x1 = self.zoom_rect[0]
        x2 = self.zoom_rect[1]
        y1 = self.zoom_rect[2]
        y2 = self.zoom_rect[3]

        # Aplicando Zoom
        self.zoomed_frame = original_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]

        # Redimensionar mantendo proporção
        height, width = self.zoomed_frame.shape[:2]
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.ratio = max(width/canvas_width, height/canvas_height)
        new_width = int(width / self.ratio)
        new_height = int(height / self.ratio)

        self.image_offset = ((canvas_width - new_width) / 2, (canvas_height - new_height) / 2)
        
        frame_resized = cv2.resize(self.zoomed_frame, (new_width, new_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

        # Mostrar na canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                            image=self.photo, anchor=tk.CENTER)

        return self.zoomed_frame.copy()

    def is_zoomed(self) -> bool:
        x1 = self.zoom_rect[0]
        x2 = self.zoom_rect[1]
        y1 = self.zoom_rect[2]
        y2 = self.zoom_rect[3]

        if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
            return False
        return True

    # -------- Funções de filtros ---------
    def apply_blur(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                self.current_frame = cv2.GaussianBlur(self.current_frame, (5,5), 0)
                self.show_frame()
                
        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('blur')
                if self.is_paused:
                    self.update_video_frame()

    
    def apply_sharpen(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                self.current_frame = cv2.filter2D(self.current_frame, -1, kernel)
                self.show_frame()
                
        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('sharpen')
                if self.is_paused:
                    self.update_video_frame()
    
    def apply_emboss(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
                self.current_frame = cv2.filter2D(self.current_frame, -1, kernel)
                self.show_frame()
        
        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('emboss')
                if self.is_paused:
                    self.update_video_frame()
    
    def apply_laplacian(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                self.current_frame = cv2.Laplacian(self.current_frame, cv2.CV_64F).astype(np.uint8)
                self.show_frame()

        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                # Loop para ler todos os quadros enquanto o vídeo está sendo processado
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('laplacian')
                if self.is_paused:
                    self.update_video_frame()
    
    def apply_canny(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                self.current_frame = cv2.Canny(self.current_frame, 100, 200)
                self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
                self.show_frame()
                
        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('canny')
                if self.is_paused:
                    self.update_video_frame()
    
    def apply_sobel(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                grad_x = cv2.Sobel(self.current_frame, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(self.current_frame, cv2.CV_64F, 0, 1, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                self.current_frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                self.show_frame()
                
        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('sobel')
                if self.is_paused:
                    self.update_video_frame()
    
    # --------- Funções de cor -----------
    def convert_grayscale(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                self.current_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                self.show_frame()

        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('gray')
                if self.is_paused:
                    self.update_video_frame()
    
    def convert_binary(self):
        if self.mode_var.get() == 'image':
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                self.current_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                self.show_frame()

        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()
                self.video_filters.append('binary')
                if self.is_paused:
                    self.update_video_frame()
    
    def restore_color(self):
        if self.mode_var.get() == 'image':
            if self.original_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame
                self.current_frame = self.original_frame.copy()
                self.show_frame()

        elif self.mode_var.get() == 'video' or self.mode_var.get() == 'webcam':
            if self.cap is not None:
                self.video_filters.clear()
                if self.is_paused:
                    self.update_video_frame()
    

    def detect_objects(self):
        if self.mode_var.get() == 'image':  # Modo imagem
            if self.current_frame is not None:
                if self.processing_mode.get() == 'independent':
                    self.current_frame = self.original_frame.copy()  # Restaurar o frame original
                    
                # Realizar detecção de objetos no frame atual
                results = self.model(self.current_frame, conf=0.5)  # Ajuste a confiança conforme necessário
                annotated_frame = results[0].plot()  # Anotar o frame com os resultados
                
                self.current_frame = annotated_frame  # Atualizar o frame atual com as anotações
                self.show_frame()  # Exibir o frame anotado

        elif self.mode_var.get() == 'video':  # Modo vídeo
            if self.cap is not None:
                if self.processing_mode.get() == 'independent':
                    self.video_filters.clear()  # Limpar filtros aplicados ao vídeo
                
                self.video_filters.append('detect_objects')  # Adicionar a detecção como um filtro
                if self.is_paused:
                    # Atualizar o frame pausado com a detecção
                    if self.current_frame is not None:
                        results = self.model(self.current_frame, conf=0.5)
                        annotated_frame = results[0].plot()
                        self.current_frame = annotated_frame
                        self.update_video_frame()  # Atualizar exibição com o frame anotado


    # Controle de exclusão do objeto
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    root = ctk.CTk()
    app = VideoImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()