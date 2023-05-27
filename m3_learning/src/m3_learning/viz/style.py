def set_style(name="default"):
    """Function to implement custom default style for graphs

    Args:
        name (str, optional): style name. Defaults to "default".
    """
    if name == "default":
        try:
            import seaborn as sns

            # resetting default seaborn style
            sns.reset_orig()

            print(f"{name} set for seaborn")

        except:
            pass

        try:
            import matplotlib.pyplot as plt
            # setting default plotting params
            plt.rcParams['image.cmap'] = 'magma'
            plt.rcParams['axes.labelsize'] = 18
            plt.rcParams['xtick.labelsize'] = 16
            plt.rcParams['ytick.labelsize'] = 16
            plt.rcParams['figure.titlesize'] = 20
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['xtick.top'] = True
            plt.rcParams['ytick.right'] = True
            print(f"{name} set for matplotlib")
        except:
            pass

    if name == "printing":

        try:
            import seaborn as sns

            # resetting default seaborn style
            sns.reset_orig()

            print(f"{name} set for seaborn")

        except:
            pass

        import matplotlib.pyplot as plt
        # setting default plotting params
        plt.rcParams['image.cmap'] = 'viridis'
        plt.rcParams['axes.labelsize'] = 6
        plt.rcParams['xtick.labelsize'] = 5
        plt.rcParams['ytick.labelsize'] = 5
        plt.rcParams['figure.titlesize'] = 8
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['lines.markersize'] = .5
        plt.rcParams['axes.grid'] = False
        plt.rcParams['lines.linewidth'] = .5
        plt.rcParams['axes.linewidth'] = .5
        plt.rcParams['legend.fontsize'] = 5
        plt.rcParams['legend.loc'] = "upper left"
        plt.rcParams['legend.frameon'] = False
