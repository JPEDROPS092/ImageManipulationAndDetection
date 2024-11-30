import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import timedelta

class VideoImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Processador de Imagens e Vídeos")
        
        # Variáveis de controle
        self.current_file = None
        self.is_video = False
        self.cap = None
        self.current_frame = None
        self.original_frame = None
        self.roi_points = []
        self.drawing_roi = False
        self.video_cutpoints = []
        self.video_speed = 1.0
        self.image_offset = (0, 0)
        self.is_paused = False
        self.processing_mode = "independent"  # ou "cascade"
        
        self.setup_gui()

        self.window_width = self.root.winfo_width
        self.window_height = self.root.winfo_width
        
    def setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controles superiores
        self.setup_top_controls()
        
        # Área de visualização
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600)
        self.canvas.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Bind eventos do mouse
        self.canvas.bind("<Button-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        self.root.bind("<Configure>", self.window_resize)
        
        # Controles de filtros e operações
        self.setup_filter_controls()
        
        # Controles de vídeo
        self.setup_video_controls()
        
    def setup_top_controls(self):
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Seleção de modo
        ttk.Label(controls_frame, text="Modo:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="image")
        ttk.Radiobutton(controls_frame, text="Imagem", variable=self.mode_var, 
                       value="image", command=self.mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(controls_frame, text="Vídeo", variable=self.mode_var,
                       value="video", command=self.mode_changed).pack(side=tk.LEFT)
        
        # Botão de abrir arquivo
        ttk.Button(controls_frame, text="Abrir Arquivo", 
                  command=self.open_file).pack(side=tk.LEFT, padx=10)
        
        # Modo de processamento
        ttk.Label(controls_frame, text="Processamento:").pack(side=tk.LEFT, padx=5)
        self.processing_mode = tk.StringVar(value="independent")
        ttk.Radiobutton(controls_frame, text="Independente", 
                       variable=self.processing_mode, 
                       value="independent").pack(side=tk.LEFT)
        ttk.Radiobutton(controls_frame, text="Cascata", 
                       variable=self.processing_mode, 
                       value="cascade").pack(side=tk.LEFT)
    
    def setup_filter_controls(self):
        filter_frame = ttk.LabelFrame(self.main_frame, text="Filtros e Operações")
        filter_frame.grid(row=2, column=0, pady=5, sticky=tk.W)
        
        # Botões de filtros
        filters = [
            ("Blur", self.apply_blur),
            ("Sharpen", self.apply_sharpen),
            ("Emboss", self.apply_emboss),
            ("Laplacian", self.apply_laplacian),
            ("Canny", self.apply_canny),
            ("Sobel", self.apply_sobel)
        ]
        
        for i, (text, command) in enumerate(filters):
            ttk.Button(filter_frame, text=text, 
                      command=command).grid(row=i//3, column=i%3, padx=5, pady=2)
        
        # Controles de cor
        color_frame = ttk.Frame(filter_frame)
        color_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(color_frame, text="Tons de Cinza", 
                  command=self.convert_grayscale).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Binário", 
                  command=self.convert_binary).pack(side=tk.LEFT, padx=5)
        ttk.Button(color_frame, text="Colorido", 
                  command=self.restore_color).pack(side=tk.LEFT, padx=5)
    
    def setup_video_controls(self):
        video_frame = ttk.LabelFrame(self.main_frame, text="Controles de Vídeo")
        video_frame.grid(row=2, column=1, pady=5, sticky=tk.W)
        
        # Controles de velocidade
        ttk.Button(video_frame, text="Acelerar", 
                  command=self.speed_up).grid(row=0, column=0, padx=5)
        ttk.Button(video_frame, text="Pausar", 
                  command=self.toggle_pause).grid(row=0, column=1, padx=5)
        ttk.Button(video_frame, text="Desacelerar", 
                  command=self.slow_down).grid(row=0, column=2, padx=5)
        
        # Controle de direção
        ttk.Button(video_frame, text="Inverter Direção", 
                  command=self.toggle_direction).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Marcação de pontos de corte
        ttk.Button(video_frame, text="Marcar Ponto de Corte", 
                  command=self.mark_cutpoint).grid(row=1, column=2, pady=5)
        
    # Funções controle da janela
    def window_resize(self, event):
        print(event.widget.widgetName)
        if event.widget.widgetName == "toplevel":
            if (self.window_width != event.width) and (self.window_height != event.height):
                self.window_width, self.window_height = event.width, event.height
                print(self.window_width, self.window_height)
    
    def mode_changed(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.canvas.delete("all")
    
    def open_file(self):
        if self.mode_var.get() == "image":
            filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        else:
            filetypes = [("Video files", "*.mp4 *.avi *.mov")]
        
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.current_file = filename
            if self.mode_var.get() == "video":
                self.open_video()
            else:
                self.open_image()
    
    def open_image(self):
        self.current_frame = cv2.imread(self.current_file)
        self.original_frame = self.current_frame.copy()
        self.show_frame()
    
    def open_video(self):
        self.cap = cv2.VideoCapture(self.current_file)
        self.update_video_frame()
    
    def update_video_frame(self):
        if self.cap is not None and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.show_frame()
            self.root.after(30, self.update_video_frame)
    
    def show_frame(self):
        if self.current_frame is not None:
            # Converter BGR para RGB
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar mantendo proporção
            height, width = frame_rgb.shape[:2]
            canvas_width = 800
            canvas_height = 600
            
            self.ratio = max(width/canvas_width, height/canvas_height)
            new_width = int(width / self.ratio)
            new_height = int(height / self.ratio)

            self.image_offset = ((canvas_width - new_width) / 2, (canvas_height - new_height) / 2)
            
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Converter para PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
            
            # Mostrar na canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.photo, anchor=tk.CENTER)
    
    # Funções de filtros
    def apply_blur(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            self.current_frame = cv2.GaussianBlur(self.current_frame, (5,5), 0)
            self.show_frame()
    
    def apply_sharpen(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.current_frame = cv2.filter2D(self.current_frame, -1, kernel)
            self.show_frame()
    
    def apply_emboss(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
            self.current_frame = cv2.filter2D(self.current_frame, -1, kernel)
            self.show_frame()
    
    def apply_laplacian(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            self.current_frame = cv2.Laplacian(self.current_frame, cv2.CV_64F).astype(np.uint8)
            self.show_frame()
    
    def apply_canny(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            self.current_frame = cv2.Canny(self.current_frame, 100, 200)
            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
            self.show_frame()
    
    def apply_sobel(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            grad_x = cv2.Sobel(self.current_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(self.current_frame, cv2.CV_64F, 0, 1, ksize=3)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            self.current_frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            self.show_frame()
    
    # Funções de cor
    def convert_grayscale(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            self.current_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.show_frame()
    
    def convert_binary(self):
        if self.current_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.current_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.show_frame()
    
    def restore_color(self):
        if self.original_frame is not None:
            if self.processing_mode.get() == 'independent':
                self.current_frame = self.original_frame
            self.current_frame = self.original_frame.copy()
            self.show_frame()
    
    # Funções de ROI
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
    
    def process_roi(self):
        if len(self.roi_points) == 2:
            # Converter coordenadas do canvas para coordenadas da imagem
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]

            x1 = int((x1 - self.image_offset[0]) * self.ratio)
            x2 = int((x2 - self.image_offset[0]) * self.ratio)
            y1 = int((y1 - self.image_offset[1]) * self.ratio)
            y2 = int((y2 - self.image_offset[1]) * self.ratio)

            x1 = max(0, x1)
            y1 = max(0, y1)
            
            # Extrair ROI
            roi = self.current_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]

            roi_ = cv2.resize(roi, (max(1, int(roi.shape[1] / self.ratio)), max(1, int(roi.shape[0] / self.ratio))))
            
            # Mostrar ROI em nova janela
            cv2.imshow("Region of Interest", roi_)
            
            # Opção de salvar
            if messagebox.askyesno("Salvar ROI", "Deseja salvar a região selecionada?"):
                filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                      filetypes=[("PNG files", "*.png")])
                if filename:
                    cv2.imwrite(filename, roi)
    
    # Funções de controle de vídeo
    def speed_up(self):
        self.video_speed *= 1.5
    
    def slow_down(self):
        self.video_speed *= 0.75
    
    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.update_video_frame()
    
    def toggle_direction(self):
        if self.cap is not None:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)
    
    def mark_cutpoint(self):
        if self.cap is not None:
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.video_cutpoints.append(current_time)
            messagebox.showinfo("Ponto Marcado", 
                              f"Tempo marcado: {timedelta(seconds=int(current_time))}")
    
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
                    
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_path, frame)
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
                    out.write(frame)
                
                out.release()
        
        messagebox.showinfo("Concluído", "Segmentos salvos com sucesso!")
        self.video_cutpoints = []  # Limpar pontos de corte
    
    def apply_zoom(self, factor):
        if self.current_frame is not None and len(self.roi_points) == 2:
            x1, y1 = self.roi_points[0]
            x2, y2 = self.roi_points[1]
            
            # Extrair ROI
            roi = self.current_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
            
            # Calcular novas dimensões
            height, width = roi.shape[:2]
            new_height = int(height * factor)
            new_width = int(width * factor)
            
            # Redimensionar
            zoomed = cv2.resize(roi, (new_width, new_height), 
                              interpolation=cv2.INTER_LINEAR if factor > 1 else cv2.INTER_AREA)
            
            # Mostrar resultado
            cv2.imshow("Zoomed Region", zoomed)
            
            # Opção de salvar
            if messagebox.askyesno("Salvar Zoom", "Deseja salvar a região com zoom?"):
                filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                      filetypes=[("PNG files", "*.png")])
                if filename:
                    cv2.imwrite(filename, zoomed)
    
    def setup_zoom_controls(self):
        zoom_frame = ttk.LabelFrame(self.main_frame, text="Controles de Zoom")
        zoom_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(zoom_frame, text="Zoom +", 
                  command=lambda: self.apply_zoom(2.0)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom -", 
                  command=lambda: self.apply_zoom(0.5)).pack(side=tk.LEFT, padx=5)
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = VideoImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()