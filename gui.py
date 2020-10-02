try:
    import tkinter.filedialog
    import tkinter as tk
    from tkinter import ttk
    import main
    import os
    import shutil
    import sklearn
    import numpy
    import matplotlib.pyplot
    import scipy
    from PIL import Image, ImageTk
except:
    import install_requirements

    import tkinter.filedialog
    import tkinter as tk
    from tkinter import ttk
    import main
    import os
    import shutil
    import sklearn
    import numpy
    import matplotlib.pyplot
    import scipy
    from PIL import Image, ImageTk

root = tk.Tk()
root.title('Free Throw Assistant')
root.resizable(0, 0)
root.iconbitmap('extras/icon.ico')
bg_color = '#253E53'
canvas = tk.Canvas(root, height=600, width=1100, bg=bg_color)
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)


def printErrorMessage(text='Please open the Free Throw shots dataset file first.'):
    tk.messagebox.showerror(title='Error', message=text)


def visualizeDataset(X, y):
    if X.shape == (1, 1):
        printErrorMessage()
        return
    fig, ax = main.visualizeDataset(X, y)
    return fig, ax

def visualizeDecisionBoundary(classifier, X, y, y_with_swishes):
    if X.shape == (1, 1):
        printErrorMessage()
        return
    fig, ax = main.visualizeDecisionBoundary(classifier, X, y, y_with_swishes)

    return fig, ax


def visualizeColormap(X, y, y_with_swishes, num_pts, y_initial):
    if X.shape == (1, 1):
        printErrorMessage()
        return
    fig, ax = main.visualizeColormap(X, y, y_with_swishes, num_pts, y_initial)

    return fig, ax


def printOptimalParameters(X, y, y_with_swishes, mu, std):
    if X.shape == (1, 1):
        printErrorMessage()
        return
    optimal_parameters, probability = main.optimalParameters(X, y, y_with_swishes, mu, std)

    text = 'Optimal Parameters\nBody-Arm angle: {:.2f}\nElbow angle: {:.2f}\n\nFree Throw percentage\nat those parameters: {:.2f}%'
    optimal_parameters_message = tk.Message(frame, text=text.format(optimal_parameters[0], optimal_parameters[1], probability[0][1] * 100),
                                        justify='left', width=200, fg='white', bg=bg_color)
    optimal_parameters_message.place(relx=0.08, rely=0.765)


def quitProtocol():
    if os.path.exists('extras/Plots'):
        shutil.rmtree('extras/Plots')
    root.quit()

def forward_button(image_number):
    global image_list, image_label, button_forward, button_back
    image_label.place_forget()
    image_label = tk.Label(frame, image=image_list[image_number - 1])
    image_label.image = image_list[image_number - 1]
    image_label.place(relx=0.31, rely=0.02)
    button_back = tk.Button(frame, text='<<', fg='white', bg=bg_color, width=5, command=lambda: back_button(image_number - 1))
    button_forward = tk.Button(frame, text='>>', fg='white', bg=bg_color, width=5, command=lambda: forward_button(image_number + 1))

    if image_number == len(image_list):
        button_forward = tk.Button(frame, text='>>', fg='white', bg=bg_color, width=5, state=tk.DISABLED)

    button_back.place(relx=0.53, rely=0.95)
    button_forward.place(relx=0.735, rely=0.95)


def back_button(image_number):
    global image_list, image_label, button_forward, button_back
    image_label.place_forget()
    image_label = tk.Label(frame, image=image_list[image_number - 1])
    image_label.image = image_list[image_number - 1]
    image_label.place(relx=0.31, rely=0.02)
    button_back = tk.Button(frame, text='<<', fg='white', bg=bg_color, width=5,
                            command=lambda: back_button(image_number - 1))
    button_forward = tk.Button(frame, text='>>', fg='white', bg=bg_color, width=5,
                               command=lambda: forward_button(image_number + 1))

    if image_number == 1:
        button_back = tk.Button(frame, text='<<', fg='white', bg=bg_color, width=5, state=tk.DISABLED)

    button_back.place(relx=0.53, rely=0.95)
    button_forward.place(relx=0.735, rely=0.95)


def openFileDialog():
    filename = tk.filedialog.askopenfilename(initialdir="/", title='Select Dataset File',
                                          filetypes=(('Text Files', '*.txt'), ('CSV Files', '*.csv')))

    if filename == '':
        return

    try:
        X, y = main.loadDataset(filename)
    except:
        printErrorMessage('The selected file does follow the format specified in the README file.')
        return

    X_norm, y_norm, mu, std, y_with_swishes = main.normalizeFeatures_addConvexHullPoints(X, y)
    classifier = main.trainModel(X_norm, y_norm)

    if os.path.exists('extras/Plots'):
        shutil.rmtree('extras/Plots')
    if not os.path.exists('extras/Plots'):
        os.makedirs('extras/Plots')

    progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=160, mode='determinate')
    progress.place(relx=0.08, rely=0.3)


    fig, ax = visualizeDataset(X, y)
    fig.savefig('extras/Plots/1.png')
    fig.clear()
    progress['value'] = 25
    root.update_idletasks()

    fig, ax = visualizeDecisionBoundary(classifier, X_norm, len(X), y)
    fig.savefig('extras/Plots/2.png')
    fig.clear()
    progress['value'] = 50
    root.update_idletasks()

    fig, ax = visualizeColormap(X_norm, y_norm, y_with_swishes, len(X), y)
    fig.savefig('extras/Plots/3.png')
    fig.clear()
    progress['value'] = 75
    root.update_idletasks()

    printOptimalParameters(X_norm, y_norm, y_with_swishes, mu, std)
    progress['value'] = 100
    root.update_idletasks()

    progress.destroy()

    image1 = ImageTk.PhotoImage(Image.open('extras/Plots/1.png').resize((667, 500)))
    image2 = ImageTk.PhotoImage(Image.open('extras/Plots/2.png').resize((667, 500)))
    image3 = ImageTk.PhotoImage(Image.open('extras/Plots/3.png').resize((667, 500)))

    global image_list
    image_list = [image1, image2, image3]

    global image_label

    first_page_image.destroy()
    image_label = tk.Label(frame, image=image1)
    image_label.image = image1
    image_label.place(relx=0.31, rely=0.02)

    global button_back, button_forward
    button_back = tk.Button(frame, text='<<', fg='white', bg=bg_color, width=5, state=tk.DISABLED)
    button_forward = tk.Button(frame, text='>>', fg='white', bg=bg_color, width=5, command=lambda: forward_button(2))
    button_back.place(relx=0.53, rely=0.95)
    button_forward.place(relx=0.735, rely=0.95)


image = ImageTk.PhotoImage(Image.open('extras/first_page.jpg').resize((667, 500)))
first_page_image = tk.Label(frame, image=image)
first_page_image.imge = image
first_page_image.place(relx=0.31, rely=0.02)

openFile = tk.Button(frame, text='Open File', padx=10, pady=5, fg='white', bg=bg_color,
                     command=openFileDialog, width=19)
openFile.place(relx=0.08, rely=0.2)


root.protocol("WM_DELETE_WINDOW", quitProtocol)
root.mainloop()