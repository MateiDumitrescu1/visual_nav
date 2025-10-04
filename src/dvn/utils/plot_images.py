import matplotlib.pyplot as plt

# display an image with matplotlib, or multiple images in a grid, with nice spacing
def plot_1_image(image, title=None, figsize=(10, 10)):
    """
    Display a single image using matplotlib.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()
    
    
def plot_n_m_image(image_matrix_pil, title=None, figsize=(10, 10)):
    """
    Display a grid of images using matplotlib.
    """
    n = len(image_matrix_pil)
    m = len(image_matrix_pil[0])
    fig, axs = plt.subplots(n, m, figsize=figsize)
    for i in range(n):
        for j in range(m):
            axs[i, j].imshow(image_matrix_pil[i][j])
            axs[i, j].axis('off')
    if title: plt.suptitle(title)
    plt.show()
