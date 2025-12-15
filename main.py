""" Etiquetador de Imágenes con OpenCV. """

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_VIDEOIO_PRIORITY_QT'] = '0'

import cv2
import json
import numpy as np
from datetime import datetime
import glob

# Variable global para el estado del mouse (necesario para Qt)
mouse_state = {
    'drawing': False,
    'start_point': None,
    'end_point': None,
    'temp_box': None,
    'original_size': None,
    'labeler': None
}

def mouse_callback_global(event, x, y, flags, param):
    """Callback global para eventos del mouse (requerido por Qt)."""
    labeler = mouse_state['labeler']
    if labeler is None:
        return
    
    original_size = mouse_state['original_size']
    if original_size is None:
        return
    
    # Limitar y al área de la imagen
    if y >= labeler.display_size[1]:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state['drawing'] = True
        mouse_state['start_point'] = (x, y)
        mouse_state['end_point'] = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and mouse_state['drawing']:
        mouse_state['end_point'] = (x, y)
        mouse_state['temp_box'] = (mouse_state['start_point'], mouse_state['end_point'])
        labeler.temp_box = mouse_state['temp_box']
    
    elif event == cv2.EVENT_LBUTTONUP and mouse_state['drawing']:
        mouse_state['drawing'] = False
        mouse_state['end_point'] = (x, y)
        
        start_point = mouse_state['start_point']
        end_point = mouse_state['end_point']
        
        # Calcular bounding box en coordenadas originales
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])
        
        # Solo agregar si el box tiene tamaño significativo
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            # Convertir a coordenadas originales
            orig_p1 = labeler._scale_point((x1, y1), original_size, labeler.display_size)
            orig_p2 = labeler._scale_point((x2, y2), original_size, labeler.display_size)
            
            # Formato [x, y, width, height]
            bbox = [
                orig_p1[0],
                orig_p1[1],
                orig_p2[0] - orig_p1[0],
                orig_p2[1] - orig_p1[1]
            ]
            
            # Agregar anotación
            img_path = labeler.images[labeler.current_idx]['path']
            if img_path not in labeler.annotations:
                labeler.annotations[img_path] = []
            
            labeler.annotations[img_path].append({
                'class': labeler.classes[labeler.current_class_idx],
                'bbox': bbox
            })
            
            print(f"Box agregado: {labeler.classes[labeler.current_class_idx]} - {bbox}")
        
        mouse_state['temp_box'] = None
        labeler.temp_box = None
        mouse_state['start_point'] = None
        mouse_state['end_point'] = None


class ImageLabeler:
    """Herramienta interactiva para etiquetar imágenes con OpenCV."""
    
    def __init__(self, data_dir, output_file="annotations.json"):
        """ Inicializa el etiquetador. """
        self.data_dir = data_dir # Directorio raíz con las carpetas de imágenes por clase
        self.output_file = os.path.join(data_dir, output_file) #  Archivo JSON donde guardar las anotaciones
        
        # Detectar clases desde las carpetas
        self.classes = self._detect_classes()
        print(f"Clases detectadas: {self.classes}")
        
        # Cargar todas las imágenes
        self.images = self._load_images()
        print(f"Total de imágenes: {len(self.images)}")
        
        # Estado actual
        self.current_idx = 0
        self.current_class_idx = 0
        self.annotations = {}
        
        # Estado del dibujo
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_box = None
        
        # Estado de la ayuda (controles)
        self.show_help = False
        
        # Configuración de visualización
        self.window_name = "Etiquetador"
        self.display_size = (1280, 720)  # Tamaño de ventana
        
        # Colores para cada clase (BGR)
        np.random.seed(42)
        self.colors = {
            cls: tuple(map(int, np.random.randint(0, 255, 3)))
            for cls in self.classes
        }
        
        # Cargar anotaciones existentes si existen
        self._load_annotations()
        
        # Mapeo de teclas a acciones
        self._key_actions = {
            ord(' '): self._next_image,
            ord('p'): self._prev_image,
            ord('s'): self._save_annotations,
            ord('r'): self._reset_current_image,
            ord('u'): self._undo_last_box,
            ord('d'): self._undo_last_box,
            ord('c'): self._change_class,
            ord('x'): self._reset_all_annotations,
            9: self._toggle_help,  # TAB
        }
    
    def _next_image(self):
        """Avanza a la siguiente imagen."""
        self.current_idx = min(self.current_idx + 1, len(self.images) - 1)
    
    def _prev_image(self):
        """Retrocede a la imagen anterior."""
        self.current_idx = max(self.current_idx - 1, 0)
    
    def _reset_current_image(self):
        """Elimina las anotaciones de la imagen actual."""
        img_path = self.images[self.current_idx]['path']
        if img_path in self.annotations:
            del self.annotations[img_path]
            print("Anotaciones de imagen actual eliminadas")
    
    def _undo_last_box(self):
        """Elimina el último bounding box agregado."""
        img_path = self.images[self.current_idx]['path']
        if img_path in self.annotations and self.annotations[img_path]:
            removed = self.annotations[img_path].pop()
            print(f"Box eliminado: {removed['class']}")
            if not self.annotations[img_path]:
                del self.annotations[img_path]
    
    def _change_class(self):
        """Cambia a la siguiente clase."""
        self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
        print(f"Clase actual: {self.classes[self.current_class_idx]}")
    
    def _reset_all_annotations(self):
        """Elimina todas las anotaciones."""
        self.annotations = {}
        print("TODAS las anotaciones eliminadas")
    
    def _toggle_help(self):
        """Muestra/oculta el panel de ayuda."""
        self.show_help = not self.show_help
    
    def _detect_classes(self):
        """Detecta las clases desde las subcarpetas."""
        classes = []
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Verificar que contiene imágenes
                images = glob.glob(os.path.join(item_path, "*.jpg")) + \
                         glob.glob(os.path.join(item_path, "*.jpeg")) + \
                         glob.glob(os.path.join(item_path, "*.png"))
                if images:
                    classes.append(item)
        return classes
    
    def _load_images(self):
        """Carga la lista de todas las imágenes."""
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in glob.glob(os.path.join(class_dir, ext)):
                    images.append({
                        'path': img_path,
                        'class': class_name,
                        'filename': os.path.basename(img_path)
                    })
        return sorted(images, key=lambda x: x['path'])
    
    def _load_annotations(self):
        """Carga anotaciones existentes."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', {})
                    # Encontrar el índice de la última imagen anotada
                    if self.annotations:
                        last_annotated = max(self.annotations.keys())
                        for i, img in enumerate(self.images):
                            if img['path'] == last_annotated:
                                self.current_idx = min(i + 1, len(self.images) - 1)
                                break
                print(f"Cargadas {len(self.annotations)} anotaciones existentes")
                print(f"Continuando desde imagen {self.current_idx + 1}")
            except Exception as e:
                print(f"Error cargando anotaciones: {e}")
                self.annotations = {}
        else:
            self.annotations = {}
    
    def _save_annotations(self):
        """Guarda las anotaciones en formato JSON."""
        # Crear estructura COCO-like
        coco_format = {
            'info': {
                'description': 'Dataset etiquetado para EfficientDet',
                'date_created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'categories': [
                {'id': i, 'name': cls} for i, cls in enumerate(self.classes)
            ],
            'images': [],
            'annotations': []
        }
        
        annotation_id = 0
        for img_id, (img_path, boxes) in enumerate(self.annotations.items()):
            # Obtener información de la imagen
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                else:
                    continue
            else:
                continue
            
            coco_format['images'].append({
                'id': img_id,
                'file_name': img_path,
                'width': w,
                'height': h
            })
            
            for box in boxes:
                coco_format['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': self.classes.index(box['class']),
                    'bbox': box['bbox'],  # [x, y, width, height]
                    'area': box['bbox'][2] * box['bbox'][3],
                    'iscrowd': 0
                })
                annotation_id += 1
        
        # Guardar también formato simple para uso directo
        simple_format = {
            'classes': self.classes,
            'annotations': self.annotations,
            'coco_format': coco_format
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(simple_format, f, indent=2)
        
        print(f"\nAnotaciones guardadas en: {self.output_file}")
        print(f"Total imágenes anotadas: {len(self.annotations)}")
        total_boxes = sum(len(boxes) for boxes in self.annotations.values())
        print(f"Total bounding boxes: {total_boxes}")
    
    def _get_current_image(self):
        """Obtiene la imagen actual."""
        if 0 <= self.current_idx < len(self.images):
            img_info = self.images[self.current_idx]
            img = cv2.imread(img_info['path'])
            return img, img_info
        return None, None
    
    def _scale_point(self, point, original_size, display_size):
        """Escala un punto de coordenadas de display a original."""
        scale_x = original_size[1] / display_size[0]
        scale_y = original_size[0] / display_size[1]
        return (int(point[0] * scale_x), int(point[1] * scale_y))
    
    def _scale_point_to_display(self, point, original_size, display_size):
        """Escala un punto de coordenadas originales a display."""
        scale_x = display_size[0] / original_size[1]
        scale_y = display_size[1] / original_size[0]
        return (int(point[0] * scale_x), int(point[1] * scale_y))
    
    def _draw_interface(self, img, img_info):
        """Dibuja la interfaz de usuario sobre la imagen."""
        original_size = img.shape[:2]
        
        # Redimensionar imagen para display
        display_img = cv2.resize(img, self.display_size)
        
        # Obtener anotaciones existentes para esta imagen
        img_path = img_info['path']
        existing_boxes = self.annotations.get(img_path, [])
        
        # Dibujar bounding boxes existentes
        for i, box in enumerate(existing_boxes):
            bbox = box['bbox']
            class_name = box['class']
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Convertir a coordenadas de display
            x1, y1 = self._scale_point_to_display((bbox[0], bbox[1]), original_size, self.display_size)
            x2, y2 = self._scale_point_to_display((bbox[0] + bbox[2], bbox[1] + bbox[3]), original_size, self.display_size)
            
            # Dibujar rectángulo
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con fondo
            label = f"{class_name} ({i+1})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(display_img, (x1, y1 - 20), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(display_img, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dibujar bounding box temporal mientras se dibuja
        if self.temp_box is not None:
            cv2.rectangle(display_img, self.temp_box[0], self.temp_box[1], (0, 255, 255), 2)
        
        # Panel de información
        panel_height = 120
        panel = np.zeros((panel_height, self.display_size[0], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Información de imagen
        info_text = [
            f"Imagen: {self.current_idx + 1}/{len(self.images)} - {img_info['filename']}",
            f"Clase sugerida: {img_info['class']} | Clase actual: {self.classes[self.current_class_idx]}",
            f"Boxes en esta imagen: {len(existing_boxes)}",
            "Controles: Click+Arrastrar=Box | n/ESPACIO=Sig | p=Ant | s=Guardar | r=Reset | u=Deshacer | q=Salir"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(panel, text, (10, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar clases disponibles
        class_panel = np.zeros((40, self.display_size[0], 3), dtype=np.uint8)
        class_panel[:] = (30, 30, 30)
        x_offset = 10
        for i, cls in enumerate(self.classes):
            color = self.colors[cls]
            if i == self.current_class_idx:
                cv2.rectangle(class_panel, (x_offset - 2, 5), (x_offset + 80, 35), (255, 255, 255), 2)
            cv2.rectangle(class_panel, (x_offset, 8), (x_offset + 15, 23), color, -1)
            cv2.putText(class_panel, f"{i}:{cls[:6]}", (x_offset + 18, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            x_offset += 90
        
        # Combinar todo
        combined = np.vstack([display_img, class_panel, panel])
        
        # Mostrar panel de ayuda si está activado
        if self.show_help:
            combined = self._draw_help_overlay(combined)
        else:
            # Mostrar indicador de ayuda
            help_hint = "Presiona TAB para ver controles"
            cv2.putText(combined, help_hint, (combined.shape[1] - 280, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return combined, original_size
    
    def _draw_help_overlay(self, img):
        """Dibuja el panel de ayuda con los controles."""
        overlay = img.copy()
        
        # Dimensiones del panel de ayuda
        panel_width = 450
        panel_height = 320
        x_start = (img.shape[1] - panel_width) // 2
        y_start = (self.display_size[1] - panel_height) // 2
        
        # Fondo semi-transparente
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height), 
                     (40, 40, 40), -1)
        
        # Mezclar con transparencia
        alpha = 0.9
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Borde del panel
        cv2.rectangle(img, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height), 
                     (100, 100, 100), 2)
        
        # Título
        title = "CONTROLES"
        cv2.putText(img, title, (x_start + 170, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Línea separadora
        cv2.line(img, (x_start + 20, y_start + 45), 
                (x_start + panel_width - 20, y_start + 45), (100, 100, 100), 1)
        
        # Lista de controles
        controls = [
            ("Click + Arrastrar", "Dibujar bounding box"),
            ("ESPACIO", "Siguiente imagen"),
            ("p", "Imagen anterior"),
            ("s", "Guardar anotaciones"),
            ("r", "Resetear imagen actual"),
            ("x", "Resetear TODAS las anotaciones"),
            ("u", "Deshacer ultimo box"),
            ("d", "Eliminar ultimo box"),
            ("c", "Cambiar clase actual"),
            ("q / ESC", "Salir y guardar"),
            ("TAB", "Ocultar este panel")
        ]
        
        y_offset = y_start + 70
        for key, description in controls:
            # Tecla
            cv2.putText(img, key, (x_start + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            # Descripción
            cv2.putText(img, description, (x_start + 180, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
        
        return img
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        original_size = param
        
        # Limitar y al área de la imagen
        if y >= self.display_size[1]:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self.temp_box = (self.start_point, self.end_point)
        
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.end_point = (x, y)
            
            # Calcular bounding box en coordenadas originales
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            # Solo agregar si el box tiene tamaño significativo
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                # Convertir a coordenadas originales
                orig_p1 = self._scale_point((x1, y1), original_size, self.display_size)
                orig_p2 = self._scale_point((x2, y2), original_size, self.display_size)
                
                # Formato [x, y, width, height]
                bbox = [
                    orig_p1[0],
                    orig_p1[1],
                    orig_p2[0] - orig_p1[0],
                    orig_p2[1] - orig_p1[1]
                ]
                
                # Agregar anotación
                img_path = self.images[self.current_idx]['path']
                if img_path not in self.annotations:
                    self.annotations[img_path] = []
                
                self.annotations[img_path].append({
                    'class': self.classes[self.current_class_idx],
                    'bbox': bbox
                })
                
                print(f"Box agregado: {self.classes[self.current_class_idx]} - {bbox}")
            
            self.temp_box = None
            self.start_point = None
            self.end_point = None

    def _processKey(self, key) -> bool:
        """Procesa la tecla presionada y ejecuta la acción correspondiente."""
        if key == 255:  # Sin tecla presionada
            return True
        
        # Salir con q o ESC
        if key in [ord('q'), 27]:
            print("\nGuardando y saliendo...")
            self._save_annotations()
            return False
        
        # Buscar acción en el diccionario
        if key in self._key_actions:
            self._key_actions[key]()
        
        return True
    
    def run(self):
        """Ejecuta la interfaz de etiquetado."""
 
        # Configurar estado global del mouse
        global mouse_state
        mouse_state['labeler'] = self
        
        # Cargar primera imagen
        img, img_info = self._get_current_image()
        if img is None:
            print("No se encontraron imágenes.")
            return
        
        display, original_size = self._draw_interface(img, img_info)
        mouse_state['original_size'] = original_size
        
        # Mostrar imagen primero (esto crea la ventana implícitamente)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(100)  # Necesario para que Qt inicialice la ventana
        
        # Ahora configurar el mouse callback
        cv2.setMouseCallback(self.window_name, mouse_callback_global)

        running = True
        
        while running:
            img, img_info = self._get_current_image()
            
            if img is None:
                print("No se pudo cargar la imagen")
                break
            
            # Establecer clase sugerida basada en la carpeta
            suggested_class = img_info['class']
            if suggested_class in self.classes:
                self.current_class_idx = self.classes.index(suggested_class)
            
            display, original_size = self._draw_interface(img, img_info)
            mouse_state['original_size'] = original_size
            
            # Mostrar imagen
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            # renombrar método 
            running = self._processKey(key)
        
        cv2.destroyAllWindows()


def create_train_val_annotations(data_dir):
    """ Crea anotaciones separadas para train, val y test. """

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            print(f"\n{'='*60}")
            print(f"Etiquetando conjunto: {split.upper()}")
            print(f"{'='*60}")
            
            labeler = ImageLabeler(
                data_dir=split_dir,
                output_file="annotations.json"
            )
            labeler.run()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Etiquetador de imágenes con OpenCV para crear bounding boxes para entrenamiento de modelos de detección de objetos (EfficientDet, YOLO, etc.)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  python main.py --data data/train              Etiquetar imágenes en data/train
  python main.py --data data/val                Etiquetar imágenes en data/val
  python main.py --output mis_anotaciones.json  Guardar anotaciones con nombre personalizado
  python main.py --all                          Etiquetar train, val y test secuencialmente
'''
    )
    parser.add_argument('--data', type=str, default='data/train',
                       help='Directorio con las imágenes organizadas por clase (default: data/train)')
    parser.add_argument('--output', type=str, default='annotations.json',
                       help='Nombre del archivo de anotaciones de salida (default: annotations.json)')
    parser.add_argument('--all', action='store_true',
                       help='Etiquetar train, val y test secuencialmente')
    
    args = parser.parse_args()
    
    if args.all:
        create_train_val_annotations('data')
    else:
        # Verificar que el directorio existe
        if not os.path.exists(args.data):
            print(f"Error: El directorio {args.data} no existe")
        else:
            labeler = ImageLabeler(
                data_dir=args.data,
                output_file=args.output
            )
            labeler.run()
