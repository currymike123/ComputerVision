import numpy as np
import cv2
import matplotlib.pyplot as plt

class InteractiveCurve:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_lut = np.arange(256)
        self.lut = self.original_lut.copy()

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.curve, = self.axs[0].plot(self.lut, self.lut, '-o', color='blue')
        self.display = self.axs[1].imshow(self.image)
        
        self.dragging = False
        self.last_idx = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_click(self, event):
        if event.inaxes != self.curve.axes:
            return
        idx = np.argmin(np.abs(self.curve.get_xdata() - event.xdata))
        if abs(self.curve.get_ydata()[idx] - event.ydata) < 10:  # threshold to select the point
            self.dragging = True
            self.last_idx = idx

    def on_release(self, event):
        self.dragging = False
        self.last_idx = None

    def on_motion(self, event):
        if not self.dragging:
            return
        self.lut[self.last_idx] = int(event.ydata)
        self.update()

    def update(self):
        # Redraw curve
        self.curve.set_ydata(self.lut)
        self.curve.figure.canvas.draw()

        # Apply LUT to the image
        adjusted_image = cv2.LUT(self.image, self.lut.astype(np.uint8))
        self.display.set_array(adjusted_image)
        self.axs[1].figure.canvas.draw()

if __name__ == "__main__":
    curve_editor = InteractiveCurve("Graphics/face.png")
    plt.show()
