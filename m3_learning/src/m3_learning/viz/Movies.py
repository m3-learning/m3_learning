import glob as glob
import cv2
from m3_learning.util.file_IO import make_folder


def make_movie(movie_name, input_folder, output_folder, file_format,
               fps, output_format='mp4', reverse=False):
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

    # reads the first image to get the shape
    img = cv2.imread(new_list[0])
    shape_ = (img.shape[1], img.shape[0])

    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        f"{output_folder}/{movie_name}.{output_format}", fourcc, fps, shape_)

    # Add frames to the video
    for image in new_list:
        frame = cv2.imread(image)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    # Release the resources
    cv2.destroyAllWindows()
