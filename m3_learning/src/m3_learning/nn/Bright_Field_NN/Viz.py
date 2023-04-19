import matplotlib.pyplot as plt
from m3_learning.viz.layout import imagemap, add_scalebar

class Viz:

    def __init__(self,
                  dset,
                 channels=None,
                 color_map='viridis',
                 printer=None,
                 labelfigs_=False,
                 scalebar_=None,
                 ):
        """Initialization of the Viz class
        """

        self.printer = printer
        self.labelfigs_ = labelfigs_
        self.scalebar_ = scalebar_
        self.cmap = plt.get_cmap(color_map)
        self.channels = channels
        self.dset = dset


    def view_raw(self, img_name):
        data = self.dset.get_img(*img_name)
        fig,axs = plt.subplots(figsize=(1.25,1.25))

        imagemap(axs, data, divider_ = True)

        axs.set_box_aspect(1)

        if self.scalebar_ is not None:
            # adds a scalebar to the figure
            add_scalebar(axs, self.scalebar_)

        if self.printer is not None:
              self.printer.savefig(fig,
                                     f'{img_name[0]}_{img_name[1]}_maps', tight_layout=False)


        


# TODO:
# widget to see all temps for 1) raw 2) filtered 3) unfiltered