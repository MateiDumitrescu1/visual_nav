import matplotlib.pyplot as plt

def plot_1_image(image, title=None, figsize=(10, 10), tight_layout=False, **kwargs):
    """
    Display a single image using `matplotlib`.

    Args:
        image: The image to display
        title: Optional title for the plot
        figsize: Figure size as (width, height)
        tight_layout: If True, apply tight_layout before showing
        **kwargs: Additional keyword arguments passed to plt.imshow
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, **kwargs)
    if title: plt.title(title)
    plt.axis('off')
    if tight_layout: plt.tight_layout()
    plt.show()
    
# display multiple images in a grid, with nice spacing
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
