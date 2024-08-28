import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import yaml
import os
import pandas as pd
from DistribuGen import DistribuGen
from SnrGen import SnrGen
from BackgroundGen_par_stack import BackgroundGen
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import ast


# Define a converter function
def str_to_tuple(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


class ImageGenPar:
    """
    Generates Image stacks in parallel, specified over a certain range of SNRs,
    saves dataset in train val test according to classes
    """

    def __init__(self, configs):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Print rank and size to confirm process initialization
        print(f"Process {rank} out of {size} processes initialized")
        self.configuration = configs
        self.filenames = self.file_names()
        master_file_path = os.path.join(self.configuration['synthetic_image_directory'], self.filenames,
                                        self.configuration['master_file_name'] + self.filenames + '_' + self.configuration['sets'][0] + '.csv')
        self.mfp = master_file_path
        if os.path.exists(self.mfp):
            existed = True
        else:
            existed = False

        if rank == 0:

            # save the config options
            yaml_contents_str = yaml.dump(configs)
            with open(os.path.join(self.configuration['synthetic_image_directory'], self.filenames,
                                   'configuration.txt'), 'w') as file:
                file.write(yaml_contents_str)

            ratio = configs['real_bogus_ratio']
            if configs['sets'][0] == 'train':
                num_stacks = self.configuration['num_stacks']
            else:
                num_stacks = self.configuration['val_stacks']

            # if there is already a master file for the specified range
            if existed:
                pass
            else:
                master_file_final = pd.DataFrame(columns=self.configuration['master_file_columns'])
                master_data = pd.DataFrame(columns=self.configuration['master_file_columns'])
                # generate master and samples
                dis_gen = DistribuGen(fp_pre=configs['sbd_file_name'], fp_post=configs['horizons_file_name'],
                                      fp_final=os.path.join(configs['synthetic_image_directory'],
                                                            str(configs['num_stacks']), configs['distribution_file_name']),
                                      cnames_ssb=configs['sbd_file_columns'],
                                      cnames_horizon=configs['horizons_file_columns'],
                                      cnames_dist=configs['distribution_file_columns'], options=configs['options'],
                                      dist_range=configs['dist_range'], observer=configs['observer'])

                dis_gen.generate(num_stacks, ratio)
                master_data.loc[:, self.configuration['distribution_file_columns']] = dis_gen.samples


                # SNR
                siggen = SnrGen(configs, dis_gen.samples)
                siggen.gen_snr_file()
                master_data.loc[:, 'Expected_SNR'] = siggen.snr_data
                master_data_cut = master_data[(master_data['Expected_SNR'] < self.configuration['snr_limits'][0]) & (
                            master_data['Expected_SNR'] >= self.configuration['snr_limits'][1])]

                # Asteroid samples
                master_file_final = pd.concat([master_file_final, master_data_cut], ignore_index=True)
                dis_gen.options = dis_gen.options + 'f'
                while len(master_file_final['Expected_SNR']) < num_stacks:
                    print("Current # of Samples for SNR " + str(self.configuration['snr_limits']) + ": " + str(len(master_file_final['Expected_SNR'])))
                    dis_gen.generate(num_stacks, ratio)
                    master_data.loc[:, self.configuration['distribution_file_columns']] = dis_gen.samples

                    # SNR
                    siggen = SnrGen(configs, dis_gen.samples)
                    siggen.gen_snr_file()
                    master_data.loc[:, self.configuration['snr_file_columns']] = siggen.snr_data
                    master_data_cut = master_data[(master_data['Expected_SNR'] < self.configuration['snr_limits'][1]) & (master_data['Expected_SNR'] >= self.configuration['snr_limits'][0])]

                    master_file_final = pd.concat([master_file_final, master_data_cut], ignore_index=True)


                master_file_final.to_csv(master_file_path, sep=',', header=True, index=False)
        else:
            pass

        print(f"Process {rank} before first barrier")
        comm.Barrier()
        print(f"Process {rank} after first barrier")

        if existed:
            pass
        else:
            master_file_path = os.path.join(self.configuration['synthetic_image_directory'], self.filenames,
                                            self.configuration['master_file_name'] + self.filenames + '_' +
                                            self.configuration['sets'][0] + '.csv')
            master_file_final = self.read_master(master_file_path)
            # Choose background and get statistics
            num_stacks = len(master_file_final['Expected_SNR'])
            aigen = BackgroundGen(configs)

            stack_df = aigen.stack_generator(num_stacks)

        comm.Barrier()

        if existed:
            pass
        else:
            if rank == 0:
                master_file_final.loc[:, self.configuration['stack_file_columns']] = stack_df

                # get signal
                siggen = SnrGen(configs, master_file_final)
                signals = siggen.signal_calc_test(master_file_final)
                master_file_final.loc[:, 'Expected_Signal'] = signals
                master_file_final.to_csv(master_file_path, sep=',', header=True, index=False)

            else:
                pass

        comm.Barrier()

        if existed:
            master_file_final = self.read_master(master_file_path)
        else:
            master_file_final = self.read_master(master_file_path)
            master_file_final = self.streak_start_calc(master_file_final)

            # Generate false samples
            num_false = int(num_stacks/ self.configuration['real_bogus_ratio'] - num_stacks)
            master_false = pd.DataFrame(columns=self.configuration['master_file_columns'])
            master_false.loc[:, 'Asteroid_Present'] = [False for i in range(num_false)]
            centers = [['(0, 0)' for j in range(self.configuration['num_frames'])] for i in range(num_false)]
            master_false.loc[:, self.configuration['center_file_columns']] = centers

            configs_false = configs.copy()
            configs_false['num_stacks'] = num_false
            aigen_false = BackgroundGen(configs)
            stack_df_false = aigen_false.stack_generator(num_false)
            master_false.loc[:, self.configuration['stack_file_columns']] = stack_df_false
            master_file_final = pd.concat([master_file_final, master_false], ignore_index=True)

        if rank == 0:
            master_file_final.to_csv(master_file_path, sep=',', header=True, index=False)
        else:
            pass

        comm.Barrier()

        return

    def file_names(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        num = self.configuration['num_stacks']
        snr_low = self.configuration['snr_limits'][0]
        snr_high = self.configuration['snr_limits'][1]
        files = self.configuration['files']
        sets = self.configuration['sets']
        classes = self.configuration['classes']
        top_folder = str(num) + '_snr_' + str(snr_low) + '_' + str(snr_high)

        if rank == 0:
            for idx, file in enumerate(files):
                for jdx, set in enumerate(sets):
                    for kdx, class_i in enumerate(classes):
                        if not os.path.exists(os.path.join(self.configuration['synthetic_image_directory'], top_folder)):
                            os.mkdir(os.path.join(self.configuration['synthetic_image_directory'], top_folder))
                        if not os.path.exists(
                                os.path.join(self.configuration['synthetic_image_directory'], top_folder, file)):
                            os.mkdir(os.path.join(self.configuration['synthetic_image_directory'], top_folder, file))
                        if not os.path.exists(
                                os.path.join(self.configuration['synthetic_image_directory'], top_folder, file, set)):
                            os.mkdir(os.path.join(self.configuration['synthetic_image_directory'], top_folder, file, set))
                        if file == 'backgrounds':
                            pass
                        else:
                            if not os.path.exists(
                                    os.path.join(self.configuration['synthetic_image_directory'], top_folder, file, set, class_i)):
                                os.mkdir(
                                    os.path.join(self.configuration['synthetic_image_directory'], top_folder, file, set, class_i))
        else:
            pass

        return top_folder

    def read_master(self, master_file_path):
        columns_to_convert = self.configuration['center_file_columns']
        columns_to_convert_final = ['Stack_Crop_Start'] + columns_to_convert
        return pd.read_csv(master_file_path, sep=',', header=0, converters={col: str_to_tuple for col in columns_to_convert_final}, names=self.configuration['master_file_columns'])


    def streak_start_calc(self, master):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        max_omega = self.configuration['output_image_width'] / (self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                                                                        3600 * self.configuration['pixel_scale']))

        master.loc[master['omega'] > max_omega * 3600, 'omega'] = max_omega * 3600
        lengths = master['omega'] * self.configuration['num_frames'] * (
                self.configuration['dt'] + self.configuration['slew_time']) / (
                          3600 * self.configuration['pixel_scale'])

        # Determine chunk size
        chunk_size = len(lengths) // size
        remainder = len(lengths) % size

        # Calculate start and end indices for this process
        start = rank * chunk_size
        end = start + chunk_size
        if rank == size - 1:
            end += remainder  # Last process takes any remaining elements

        centers = []
        for idx in range(start, end):
            if master['Asteroid_Present'].iloc[idx] == False:
                center_x = 0
                center_y = 0
            else:
                center_x = self.configuration['output_image_width'] + max(lengths) + 100
                center_y = self.configuration['output_image_height'] + max(lengths) + 100
                trys = 0
                while (center_x + lengths[idx] * np.cos(np.deg2rad(master['theta'].iloc[idx])) > self.configuration['output_image_width']) and (center_y + lengths[idx] * np.sin(np.deg2rad(master['theta'].iloc[idx])) > self.configuration['output_image_height']) and (trys < self.configuration['number_of_trys']):
                    trys += 1
                    try:
                        center_x = np.random.randint(lengths[idx], self.configuration['output_image_width'] - lengths[idx])
                        center_y = np.random.randint(lengths[idx], self.configuration['output_image_height'] - lengths[idx])
                    except ValueError:
                        if master['theta'].iloc[idx] > 90:
                            if master['theta'].iloc[idx] > 180:
                                if master['theta'].iloc[idx] > 270:
                                    center_x = np.random.randint(0, self.configuration[
                                        'output_image_width'] / 8)
                                    center_y = np.random.randint(7 * self.configuration['output_image_height'] / 8,
                                                                 self.configuration['output_image_height'])
                                else:
                                    center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8,
                                                                 self.configuration['output_image_width'])
                                    center_y = np.random.randint(7 * self.configuration['output_image_height'] / 8,
                                                                 self.configuration['output_image_height'])
                            else:
                                center_x = np.random.randint(7 * self.configuration['output_image_width'] / 8, self.configuration['output_image_width'])
                                center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)
                        else:
                            center_x = np.random.randint(0, self.configuration['output_image_width'] / 8)
                            center_y = np.random.randint(0, self.configuration['output_image_height'] / 8)
            print(f"Process {rank}, Center idx {idx}")

            centers_i = self.track_centers(master.iloc[idx], center_x, center_y)
            centers.append(centers_i)

        # Gather all local centers_x and centers_y at root process (rank 0)
        all_centers = comm.gather(centers, root=0)

        if rank == 0:
            # Flatten the 3D list to a 2D list
            flattened_data = [inner_list for outer_list in all_centers for inner_list in outer_list]
            all_centers_df = pd.DataFrame(flattened_data, columns=self.configuration['center_file_columns'])
            all_centers_df.reset_index(drop=True)
            master.loc[:, self.configuration['center_file_columns']] = all_centers_df
            return master
        else:
            return


    def track_centers(self, row, center_x, center_y):

        center_x0 = center_x
        center_y0 = center_y
        centers = []
        for kdx in range(0, self.configuration['num_frames']):
            center_xp1 = center_x0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                    self.configuration['dt'] + self.configuration['slew_time']) * np.cos(np.deg2rad(row['theta']))
            center_yp1 = center_y0 + self.configuration['pixel_scale'] * row['omega'] / 3600 * kdx * (
                    self.configuration['dt'] + self.configuration['slew_time']) * np.sin(np.deg2rad(row['theta']))
            centers.append((center_xp1, center_yp1))
        return centers

    def gen_images_and_file(self):

        print("Creating Stacks")

        master = self.read_master(self.mfp)

        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Determine chunk size
        chunk_size = len(master) // size
        remainder = len(master) % size

        # Calculate start and end indices for this process
        start = rank * chunk_size
        end = start + chunk_size
        if rank == size - 1:
            end += remainder  # Last process takes any remaining rows

        back_gen = BackgroundGen(self.configuration)

        file_paths = []

        # Iterate over assigned rows
        for idx in range(start, end):
            row = master.iloc[idx]
            # Process row here (replace with your actual processing logic)
            print(f"Process {rank} processing row {idx}")

            big_l = row['omega'] * self.configuration['dt'] / (3600 * self.configuration['pixel_scale'])
            signals = []
            final_images = []
            backgrounds = []

            for jdx in range(0, self.configuration['num_frames']):

                #  open image
                background_image = back_gen.known_crop_stack(row, jdx)

                backgrounds.append(background_image)

                if row['Asteroid_Present'] == False:
                    final_image = back_gen.process(background_image, row)
                else:
                    # make meshgrid corresponding to entire image
                    x = np.linspace(0, self.configuration['output_image_width'] - 1,
                                    self.configuration['output_image_width'])
                    y = np.linspace(0, self.configuration['output_image_height'] - 1,
                                    self.configuration['output_image_height'])
                    big_x, big_y = np.meshgrid(x, y)

                    center_column = f"F{jdx + 1}_Center"
                    center_x = row[center_column][0]
                    center_y =row[center_column][1]

                    # calculate Gaussian at each pixel in image apply Jedicke
                    signal = self.gaussian_streak(big_x, big_y, row, center_x, center_y, big_l)
                    signals.append(signal)

                    if np.max(signal) > np.max(background_image):
                        signal = signal * np.max(background_image) / np.max(signal)

                    final_image = back_gen.process(
                        signal + background_image, row)

                final_images.append(final_image)

            scaled_image_array = (np.array(final_images) * 255).astype(np.uint8)
            final_image_array = np.repeat(scaled_image_array[:, :, :, np.newaxis], 3, axis=3)

            if self.configuration['output_file'] == 'array':
                stack_name = 'stack{0}'.format(idx) + '.npy'
            else:
                stack_name = 'stack{0}'.format(idx) + '.avi'
            true_path = os.path.join(self.configuration['synthetic_image_directory'], self.filenames, self.configuration['files'][0], self.configuration['sets'][0], self.configuration['classes'][0])
            false_path = os.path.join(self.configuration['synthetic_image_directory'], self.filenames, self.configuration['files'][0], self.configuration['sets'][0], self.configuration['classes'][1])
            video_file_path = true_path if row['Asteroid_Present'] == True else false_path
            file_paths.append(os.path.join(video_file_path, stack_name))
            if self.configuration['output_file'] == 'array':
                # permute to (num_frames, num_channels, height, width)
                ordered_final_image_array = np.transpose(final_image_array, (0, 3, 1, 2))
                np.save(os.path.join(video_file_path, stack_name), ordered_final_image_array)
            else:
                self.video_file(final_image_array, os.path.join(video_file_path, stack_name))

        # Synchronize processes
        comm.Barrier()

        all_file_paths = comm.gather(file_paths, root=0)

        if rank == 0:
            all_file_paths = [item for sublist in all_file_paths for item in sublist]
            master['Saved_as_Stack'] = all_file_paths
            master.to_csv(self.mfp, sep=',', header=True, index=False)
        else:
            pass

        return

    def gaussian_streak(self, x, y, row, x_0, y_0, big_l):
        if big_l == 0:
            big_l = 0.000001
        arg_1 = (-(x - x_0) * np.sin(np.deg2rad(row['theta'])) + (y - y_0) * np.cos(np.deg2rad(row['theta']))) ** 2 / (
                2 * row['sigma_g'] ** 2)
        arg_2 = ((x - x_0) * np.cos(np.deg2rad(row['theta'])) + (y - y_0) * np.sin(
            np.deg2rad(row['theta'])) + big_l / 2) / (np.sqrt(2) * row['sigma_g'])
        arg_3 = ((x - x_0) * np.cos(np.deg2rad(row['theta'])) + (y - y_0) * np.sin(
            np.deg2rad(row['theta'])) - big_l / 2) / (np.sqrt(2) * row['sigma_g'])
        arg_4 = row['Expected_Signal'] / (big_l * 2 * row['sigma_g'] * np.sqrt(2 * np.pi))
        # signal
        s_xy = arg_4 * np.exp(-arg_1) * (scipy.special.erf(arg_2) - scipy.special.erf(arg_3))
        # statistical noise
        # Generate a vector with values sampled from a normal distribution
        # with mean 0 and unit variance
        # new_vector = np.random.normal(loc=0, scale=1, size=s_xy.shape)
        return s_xy #+ new_vector * np.sqrt(s_xy)

    @staticmethod
    def video_file(video_array, name):
        # Define the codec and create VideoWriter object
        num_frames, height, width, channels = video_array.shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec used to compress the frames
        video_filename = name
        video_out = cv2.VideoWriter(video_filename, fourcc, 2.0, (width, height))

        # Write each frame to the video
        for i in range(num_frames):
            frame = video_array[i]
            video_out.write(frame)  # Write the frame

        # Release everything when job is finished
        video_out.release()

        return




if __name__ == '__main__':
    with open('config_par.yaml', 'r') as f:
        config = yaml.safe_load(f)
    im_gen_par = ImageGenPar(config)
    im_gen_par.gen_images_and_file()
