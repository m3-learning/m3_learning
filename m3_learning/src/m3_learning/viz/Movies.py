import glob as glob
import cv2
from m3_learning.util.file_IO import make_folder
from tqdm import tqdm
import numpy as np

def make_movie(movie_name, input_folder, output_folder, file_format,
               fps, output_format='mp4', reverse=False, text_list=None,pad_image=True):
    """Function that constructs a movie from images

    Args:
        movie_name (string): filename to save the movie
        input_folder (path): folder where the images are located
        output_folder (path): path where the movies will be saved
        file_format (string): format of the images to use when generating a movie
        fps (int): frames per second
        output_format (str, optional): movie file format. Defaults to 'mp4'.
        reverse (bool, optional): selects if should go in a cycle. Defaults to False.
    """

    # makes the output folder
    output_folder = make_folder(output_folder)

    # searches the folder and finds the files
    file_list = glob.glob(input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob(input_folder + '/*.' + file_format)
    list.sort(file_list_rev, reverse=True)

    # combines the file list if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list

    frames = []
    # Add frames to the list
    for i,image in enumerate(tqdm(new_list)):
        frames.append(cv2.imread(image))
        
    # get the largest shape of images
    shapes = np.array([frame.shape for frame in frames]).max(axis=0)
    shape_ = (shapes[1], shapes[0])

    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        f"{output_folder}/{movie_name}.{output_format}", fourcc, fps, shape_)

    # Add frames to the video
    for i,frame in enumerate(frames):
                
        if pad_image:
            frame = cv2.copyMakeBorder(frame, 0, shape_[1]-frame.shape[0], 0, shape_[0]-frame.shape[1], 
                                       cv2.BORDER_REPLICATE)
        else: frame = cv2.imread(image)

        # Add text
        if text_list is not None:
            disp_text = new_list[i].split('/')[-1].split(f'.{file_format}')[0]
            # describe the type of font to be used. 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(frame, disp_text,
                (50, shape_[1]-50),  
                font, 3,  
                (0,0,255),  
                2, cv2.LINE_4) 
            
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    # Release the resources
    cv2.destroyAllWindows()