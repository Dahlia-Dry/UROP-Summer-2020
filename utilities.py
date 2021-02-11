import os
import subprocess
from paramiko import SSHClient
from scp import SCPClient
import scp
from astropy.io import fits
import glob
import shutil
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import pickle
import ipyvolume as p3
import matplotlib.cm as cm
import matplotlib as mat
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import FloatSlider, ColorPicker, VBox, jslink
from scipy import stats
from astropy.visualization import ZScaleInterval, LinearStretch,ImageNormalize

class StarField(object):
    """
    Object representing an individual starfield in the mitastrometry database.
    Defined by the following arguments:
    -telescope (str) = telescope used, eg. 'SARAS'
    -date (str) = date taken, eg. '20190823'
    -number (str) = frame number, eg. '0357'
    -data (Pandas DataFrame) = .Rb data for each star in image, can be read using csv_reader helper function
    -fits_file (str) = name of .R file, does not need .fits extension
    -create (bool) = True if you want to make a corresponding directory for the starfield
    """
    def __init__(self, telescope, date, number, data, fits_file, create=False):
        self.telescope = telescope
        self.date = date
        self.number = number
        self.data = data
        self.create= create
        self.fits_file = fits_file
        self.directory = str(self.date) + '_' + str(self.number) + "_analysis"
        if not os.path.exists(self.directory):
            os.system('mkdir ' + self.directory)
        """else:
            raise Exception('Directory for ' + self.date + '_' + self.number +
                            "already exists. If you wish to perform analysis again, delete existing directory.")"""
        self.outfile = os.path.join(self.directory, self.number + 'analysis.txt')
        #self.instrument = fits.open(fits_file)[0].header['INSTRUME']
        self.object = fits.open(fits_file)[0].header['OBJECT']

    def describe(self):
        return {'telescope':self.telescope,'date': self.date,'number': self.number,
                'object':self.object}

    def write_outfile(self, mpv=False):
        if self.create:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
        if mpv:
            if not os.path.exists(os.path.join(self.directory, self.number + '_mpvs.csv')):
                mpv_list = pd.Series([self.mpv_excluding_annulus(star) for star in range(len(self.data['Eta']))],
                                     index=None)
                mpv_list.to_csv(os.path.join(self.directory, self.number + '_mpvs.csv'))
            else:
                mpv_list = pd.read_csv(os.path.join(self.directory, self.number + '_mpvs.csv'))
        n = 1
        f = open(self.outfile, 'w')
        f.write('____________________________________________________________________\n')
        print('____________________________________________________________________\n')
        f.write('STATISTICAL ANALYSIS OF ' + self.date + "_" + self.number + "\n")
        print('STATISTICAL ANALYSIS OF ' + self.date + "_" + self.number + "\n")
        f.write('telescope: ' + self.telescope + '\n')
        print('telescope: ' + self.telescope + '\n')
        #f.write('instrument: ' + str(self.instrument) + '\n')
        #print('instrument: ' + str(self.instrument) + '\n')
        f.write('____________________________________________________________________\n')
        print('____________________________________________________________________\n')
        f.write('RESIDUAL PLOTS\n')
        print('RESIDUAL PLOTS\n')

        f.write('Xi Residual Magnitudes:\n')
        print('Xi Residual Magnitudes:\n')
        ximu, xisig, r, r2 = self.plot_resids('xi', 'fig'+str(n))
        f.write('fig' + str(n) + '\t' + 'mean = ' + str(ximu) + '\t std = ' + str(xisig) + '\n')
        print('fig' + str(n) +'\t' + 'mean = ' + str(ximu) + '\t std = ' + str(xisig) + '\n')
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n + 1

        f.write('Eta Residual Magnitudes:\n')
        print('Eta Residual Magnitudes:\n')
        etamu, etasig, r, r2 = self.plot_resids('eta', 'fig' + str(n))
        f.write('fig' + str(n) +  '\t' + 'mean = ' + str(etamu) + '\t std = ' + str(etasig) + '\n')
        print('fig'  + str(n) + '\t' + 'mean = ' + str(etamu) + '\t std = ' + str(etasig) + '\n')
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n + 1

        f.write('____________________________________________________________________\n')
        print('____________________________________________________________________\n')
        f.write('UNIVARIATE ANALYSIS\n')
        print('UNIVARIATE ANALYSIS\n')

        f.write('Residual Magnitude:')
        print('Residual Magnitude:')
        mu, sig = self.univariate('resids', 'fig' + str(n))
        f.write('fig' + str(n) + '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        print('fig' + str(n) +  '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        n = n + 1

        if mpv:
            f.write('Subplot Mean Pixel Value:')
            print('Subplot Mean Pixel Value:')
            mu, sig = self.univariate('mpv', 'fig' + str(n))
            f.write('fig' + str(n) + '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
            print('fig' + str(n) +'\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
            n = n + 1

        f.write('Stellar Magnitude:')
        print('Stellar Magnitude:')
        mu, sig = self.univariate('MagAuto', 'fig' + str(n))
        f.write('fig'  + str(n) + '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        print('fig' + str(n) + '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        n = n + 1

        f.write('FWHM:')
        print('FWHM:')
        mu, sig = self.univariate('FWHM', 'fig' + str(n))
        f.write('fig' + str(n) +  '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        print('fig' + str(n) + '\t' + 'mean = ' + str(mu) + '\t std = ' + str(sig) + '\n')
        n = n + 1

        f.write('____________________________________________________________________\n')
        print('____________________________________________________________________\n')
        f.write('BIVARIATE ANALYSIS\n')
        print('BIVARIATE ANALYSIS\n')
        f.write('--Colormap overlay is a Gaussian Kernel Density Estimation--\n')
        print('--Colormap overlay is a Gaussian Kernel Density Estimation--\n')
        f.write('--KDEs are zoomed to middle 80% of data for visual clarity--\n')
        print('--KDEs are zoomed to middle 80% of data for visual clarity--\n')
        f.write('--Raw scatterplots show totality of data--\n')
        print('--Raw scatterplots show totality of data--\n')

        f.write('Residual Magnitude vs Stellar Magnitude:\n')
        print('Residual Magnitude vs Stellar Magnitude:\n')
        f.write('figs ' + str(n) + ',' + str(n+1) + ':\n')
        print('figs ' + str(n) + ',' + str(n+1) + ':\n')
        r, r2 = self.bivariate('MagAuto','resids', 'fig' + str(n), 'fig' + str(n+1))
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n+2

        if mpv:
            f.write('Residual Magnitude vs Subplot Mean Pixel Value:\n')
            print('Residual Magnitude vs Subplot Mean Pixel Value:\n')
            f.write('figs ' + str(n) + ',' + str(n+1) + ':\n')
            print('figs ' + str(n) + ',' + str(n+1) + ':\n')
            r, r2 = self.bivariate('mpv','resids', 'fig' + str(n), 'fig' + str(n+1))
            f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
            print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
            n = n + 2

        f.write('Residual Magnitude vs FWHM:\n')
        print('Residual Magnitude vs FWHM:\n')
        f.write('figs ' + str(n) + ',' + str(n+1) + ':\n')
        print('figs ' + str(n) + ',' + str(n+1) + ':\n')
        r, r2 = self.bivariate('FWHM', 'resids' ,'fig' + str(n), 'fig' + str(n+1))
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n + 2

        f.write('Stellar Magnitude vs FWHM:\n')
        print('Stellar Magnitude vs FWHM:\n')
        f.write('figs ' + str(n) + ',' + str(n + 1) + ':\n')
        print('figs ' + str(n) + ',' + str(n + 1) + ':\n')
        r, r2 = self.bivariate('MagAuto', 'FWHM', 'fig' + str(n), 'fig' + str(n + 1),diff=True)
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n + 2

        f.write('Elongation vs Ellipticity:\n')
        print('Elongation vs Ellipticity:\n')
        f.write('figs ' + str(n) + ',' + str(n + 1) + ':\n')
        print('figs ' + str(n) + ',' + str(n + 1) + ':\n')
        r, r2 = self.bivariate('Elong', 'Ellip', 'fig' + str(n), 'fig' + str(n + 1),diff=True)
        f.write('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        print('r = ' + str(r) + '\t r^2 = ' + str(r2) + '\n')
        n = n + 2

        f.write('____________________________________________________________________\n')
        print('____________________________________________________________________\n')
        f.write('MULTIVARIATE ANALYSIS\n')
        print('MULTIVARIATE ANALYSIS\n')

        f.write('3D Residual Quiver Plot: fig' + str(n) +  '\n')
        print('3D Residual Quiver Plot: fig' + str(n) + '\n')
        self.plot_3D('quiver', 'fig' + str(n))
        n = n + 1

        f.write('3D Residual Surface Triangulation: fig' + str(n) + '\n')
        print('3D Residual Surface Triangulation: fig' + str(n) + '\n')
        self.plot_3D('trisurf', 'fig' + str(n))
        n = n + 1
        f.close()

    def getdata(self, param):
        if param == "resids":
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            numbers = np.array([(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))])
        elif param == "mpv":
            numbers = np.array(pd.read_csv(os.path.join(self.directory, self.number + '_mpvs.csv'), header=None)[1])
        else:
            numbers = self.data[param]
        return numbers

    def plot_resids(self, param, fname):
        if param == 'xi':
            oxi = self.data['O-Cxi']
            xi = self.data['Xi']
            ximu = np.mean(oxi)
            xisig = np.std(oxi)
            r = np.corrcoef(oxi, xi)[0][1]
            plt.xlabel('Xi')
            plt.ylabel('O-Cxi')
            plt.title('Xi Residuals')
            plt.scatter(xi, oxi, s=1)
            plt.savefig(os.path.join(self.directory, fname + '.png'))
            plt.show()
            return ximu, xisig, r, r**2
        elif param == 'eta':
            oeta = self.data['O-Ceta']
            eta = self.data['Eta']
            etamu = np.mean(oeta)
            etasig = np.std(oeta)
            r= np.corrcoef(oeta, eta)[0][1]
            plt.xlabel('Eta')
            plt.ylabel('O-Ceta')
            plt.title('Eta Residuals')
            plt.scatter(eta, oeta, s=1, color = 'orange')
            plt.savefig(os.path.join(self.directory, fname + '.png'))
            plt.show()
            return etamu, etasig, r, r**2

    def univariate(self, param, fname):
        label = param
        if param == "resids":
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            numbers = np.array([(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))])
            label = 'Residual Magnitude'
        elif param == "mpv":
            numbers = np.array(pd.read_csv(os.path.join(self.directory, self.number + '_mpvs.csv'), header=None)[1])
            label = 'Subplot Mean Pixel Value'
        else:
            numbers = self.data[param]

        bins = int(1+3.22*np.log(len(numbers))) #Sturge's Rule
        mu = np.mean(numbers)
        sig = np.std(numbers)
        plt.hist(numbers, bins)
        plt.title(label + " Distribution")
        plt.savefig(os.path.join(self.directory, fname + '.png'))
        plt.show()
        return mu, sig

    def bivariate(self, x,y, fname, fname2, diff=False):
        xlabel = x
        ylabel = y
        if y == "resids":
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            param2_i = np.array([(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))])
            ylabel = 'Residual Magnitude'
        else:
            param2_i = np.array(self.data[y])

        if x == "mpv":
            param1_i = np.array(pd.read_csv(os.path.join(self.directory, self.number + '_mpvs.csv'), header=None)[1])
            #param1_i = [self.get_mean_subplot_value(i) for i in range(len(self.data['Eta']))]
            xlabel = 'Subplot Mean Pixel Value'
        else:
            param1_i = np.array(self.data[x])

        param1 = []
        param2 = []
        x1 = np.percentile(param1_i, 10)
        x3 = np.percentile(param1_i, 90)
        y1 = np.percentile(param2_i, 10)
        y3 = np.percentile(param2_i, 90)
        for i in range(len(param1_i)):
            if param1_i[i] > x1 and param1_i[i] < x3 and param2_i[i] > y1 and param2_i[i] < y3:
                param1.append(param1_i[i])
                param2.append(param2_i[i])
        param1 = np.array(param1)
        param2 = np.array(param2)

        xmin = param1.min()
        xmax = param1.max()
        ymin = param2.min()
        ymax = param2.max()
        xprime, yprime = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xprime.ravel(), yprime.ravel()])
        values = np.vstack([param1, param2])
        kernel = stats.gaussian_kde(values)
        z = np.reshape(kernel(positions).T, xprime.shape)
        fig = plt.figure(figsize=(18,18))
        ax = fig.add_subplot(111)
        ax.imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
        ax.plot(param1, param2, 'k.', markersize=1)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Gaussian Kernel Density Estimation for " + ylabel + " vs " + xlabel)
        plt.savefig(os.path.join(self.directory, fname + '.png'))
        plt.show()
        r = np.corrcoef(param1, param2)[0][1]
        plt.figure(2)
        if diff:
            categories = self.searchhighestresids(75, colormap = True)
            categories = np.array(categories)
            colormap = np.array(['b', 'orange'])
            plt.scatter(param1_i, param2_i, s=0.75, c = colormap[categories])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(ylabel + " vs " + xlabel + " Raw Data")
            plt.savefig(os.path.join(self.directory, fname2 + '.png'))
            plt.show()
        else:
            plt.scatter(param1_i, param2_i, s=0.75)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(ylabel + " vs " + xlabel + " Raw Data")
            plt.savefig(os.path.join(self.directory, fname2 + '.png'))
            plt.show()
        return r, r**2

    def plot_3D(self, type , fname):
        x = self.data['XWIN']
        y = self.data['YWIN']
        oceta = self.data['O-Ceta']
        oxi = self.data['O-Cxi']
        resids = [(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))]
        if type == 'trisurf':
            z = resids
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlabel('XWIN')
            ax.set_ylabel('YWIN')
            ax.set_zlabel('magnitude of residual')
            ax.plot_trisurf(x, y, z, cmap='magma')
            plt.savefig(os.path.join(self.directory, fname + '.png'))
            plt.show()

        elif type == 'quiver':
            x = np.array(x)
            y = np.array(y)
            over = self.searchhighestresids(90)
            #x = [self.data['XWIN'][i] for i in over]
            #y = [self.data['YWIN'][i] for i in over]
            z = np.array(resids)
            u = np.array(self.data['O-Ceta'])
            v = np.array(self.data['O-Cxi'])
            w = np.array([0 for i in range(len(v))])
            p3.clear()
            quiver = p3.quiver(x,y,z,u,v,w, size = 3)
            #p3.savefig(os.path.join(self.directory, fname + '.png'))
            p3.show()

    def searchhighestresids(self, percentile, colormap = False): #percentile is a number between 0 and 100
        if not colormap:
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            resids = [(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))]
            resid_thresh = np.percentile(resids, percentile)
            selected_indices = []
            for i in range(len(resids)):
                if resids[i] > resid_thresh:
                    selected_indices.append(i)
            return selected_indices
        else:
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            resids = [(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))]
            resid_thresh = np.percentile(resids, percentile)
            categories = []
            for i in range(len(resids)):
                if resids[i] > resid_thresh:
                    categories.append(1)
                else:
                    categories.append(0)
            return categories

    def getresidpercentile(self, percentile, fwhm = False):
        if not fwhm:
            oceta = self.data['O-Ceta']
            oxi = self.data['O-Cxi']
            resids = [(((oceta[i] ** 2) + (oxi[i] ** 2)) ** .5) for i in range(len(oceta))]
            resid_thresh = np.percentile(resids, percentile)
            mindelta = 1000
            winner = 0
            for i in range(len(resids)):
                if np.abs(resid_thresh - resids[i]) < mindelta:
                    mindelta = np.abs(resid_thresh - resids[i])
                    winner = i
            return winner
        else: #FWHM/MagAuto
            resids = [self.data['FWHM'][i]/self.data['MagAuto'][i] for i in range(len(self.data['FWHM']))]
            resid_thresh = np.percentile(resids, percentile)
            mindelta = 1000
            winner = 0
            for i in range(len(resids)):
                if np.abs(resid_thresh - resids[i]) < mindelta:
                    mindelta = np.abs(resid_thresh - resids[i])
                    winner = i
            return winner

    def get_mean_subplot_value(self, star):
        image_data = fits.getdata(self.fits_file)
        bound = 20
        x = self.data['XIM'][star]
        y = self.data['YIM'][star]
        if y-bound < 0:
            ylow = 0
        else:
            ylow = int(y-bound)
        if x-bound < 0:
            xlow = 0
        else:
            xlow = int(x-bound)
        if y+bound > len(image_data):
            yup = len(image_data)
        else:
            yup = int(y+bound)
        if x+bound > len(image_data[0]):
            xup = len(image_data[0])
        else:
            xup = int(x+bound)
        image_data = image_data[ylow:yup, xlow:xup]
        pixelvals = np.array(image_data.flatten())
        meanval = np.mean(pixelvals)
        return meanval

    def get_stardata(self, star):
        image_data = fits.getdata(self.fits_file)
        bound = 20
        x = self.data['XIM'][star]
        y = self.data['YIM'][star]
        # print(image_data)
        # image_data = image_data[int(y)-20: int(y)+20][int(x)-20: int(x)+20]
        ylow, yup, xlow, xup = 0, 0, 0, 0
        if y - bound < 0:
            ylow = 0
        else:
            ylow = int(y - bound)
        if x - bound < 0:
            xlow = 0
        else:
            xlow = int(x - bound)
        if y + bound > len(image_data):
            yup = len(image_data)
        else:
            yup = int(y + bound)
        if x + bound > len(image_data[0]):
            xup = len(image_data[0])
        else:
            xup = int(x + bound)
        image_data = image_data[ylow:yup, xlow:xup]
        centroid = (self.data['XIM'][star]-xlow, self.data['YIM'][star]-ylow)
        centroid2 = (self.data['XWIN'][star]-xlow, self.data['YWIN'][star]-ylow)
        annulus = self.data['FWHM'][star]

        return image_data, annulus, centroid, centroid2

    def mpv_excluding_annulus(self, star):
        image_data, annulus, centroid, centroid2 = self.get_stardata(star)
        #exclude pixel values within annulus
        nap = 0
        pixcount = 0
        for i in range(len(image_data)):
            for j in range(len(image_data[0])):
                placeholder = [[j - 0.25, i - 0.25],
                               [j - 0.25, i + 0.25],
                               [j + 0.25, i - 0.25],
                               [j + 0.25, i + 0.25]]
                for k in placeholder:
                    if annulus ** 2 < (k[0] - centroid[0]) ** 2 + (k[1] - centroid[1]) ** 2:
                        pixcount += image_data[i][j] / 4
                        nap += 0.25
        mpv = pixcount/nap
        return mpv

    def starstatgraph(self, star): #star is an index in data
        lighthist_bins = 150
        residhist_bins = 150
        maghist_bins = 150
        image_data, annulus, centroid, centroid2 = self.get_stardata(star)

        #let the plots begin
        plt.figure(1)
        fig, ax = plt.subplots()
        plt.imshow(image_data, aspect="auto")
        plt.title("Star at Index " + str(star))
        circle = plt.Circle(centroid, annulus, fill = False, color = 'red')
        ax.add_artist(circle)
        plt.colorbar()
        plt.scatter(centroid[0],centroid[1], color='red', s = 1)
        #plt.scatter(centroid2[0], centroid2[1], color = 'orange', s = 0.5)
        plt.savefig(os.path.join(self.directory, str(star) + 'a.png'))
        plt.show()

        plt.figure(2)
        if not os.path.exists(os.path.join(self.directory, self.number + '_mpvs.csv')):
            mpv = pd.Series([self.mpv_excluding_annulus(star) for star in range(len(self.data['Eta']))])
            mpv.to_csv(os.path.join(self.directory, self.number + '_mpvs.csv'))

        mpv = np.array(pd.read_csv(os.path.join(self.directory, self.number + '_mpvs.csv'), header=None)[1])
        n, bins, patches = plt.hist(mpv,lighthist_bins)
        patchindex = 0
        for i in range(1, len(bins) - 1):
            if mpv[star] > bins[i - 1] and mpv[star] < bins[i + 1]:
                patchindex = i - 1
        patches[patchindex].set_fc('r')
        plt.title("Subplot Pixel Value Distribution")
        z = (mpv[star] - np.mean(mpv)) / np.std(mpv)
        zscore = "z = " + str(z)
        px = np.max(mpv) - 0.25 * (np.max(mpv) - np.min(mpv))
        py = np.max(n) - 0.25 * (np.max(n) - np.min(n))
        plt.text(px, py, zscore, ha='center', va='center')
        plt.savefig(os.path.join(self.directory, str(star) + 'b.png'))
        plt.show()

        plt.figure(3)
        oceta = self.data['O-Ceta']
        oxi = self.data['O-Cxi']
        resids = [(((oceta[i]**2)+(oxi[i]**2))**.5) for i in range(len(oceta))]
        n, bins, patches = plt.hist(resids, bins=residhist_bins, align='left', color='g')
        patchindex = 0
        for i in range(1,len(bins)-1):
            if resids[star] > bins[i-1] and resids[star] < bins[i+1]:
                patchindex = i-1
        patches[patchindex].set_fc('r')
        plt.title("Distribution of Residual Magnitudes")
        z = (resids[star] - np.mean(resids))/np.std(resids)
        zscore = "z = " + str(z)
        px = np.max(resids) - 0.25 * (np.max(resids) - np.min(resids))
        py = np.max(n) - 0.25 * (np.max(n) - np.min(n))
        plt.text(px, py, zscore, ha='center', va='center')
        plt.savefig(os.path.join(self.directory, str(star) + 'c.png'))
        plt.show()

        plt.figure(4)
        mag = self.data['MagAuto']
        n, bins, patches = plt.hist(mag, bins=maghist_bins, align='left', color='y')
        patchindex = 0
        for i in range(1, len(bins) - 1):
            if mag[star] > bins[i - 1] and mag[star] < bins[i + 1]:
                patchindex = i - 1
        patches[patchindex].set_fc('r')
        plt.title("Distribution of Stellar Magnitudes")
        z = (mag[star] - np.mean(mag)) / np.std(mag)
        zscore = "z = " + str(z)
        px = np.min(mag) + 0.25*(np.max(mag) - np.min(mag))
        py = np.max(n) - 0.25*(np.max(n) - np.min(n))
        plt.text(px, py, zscore, ha='center', va='center')
        plt.savefig(os.path.join(self.directory, str(star) + 'd.png'))
        plt.show()


def satisfies_constraints(constraints):
    try:
        fd = open('creds.txt')
    except FileNotFoundError:
        print('Create file creds.txt in this directory with pipeline username on line 1 and password on line 2')
    username = fd.readline().strip('\n')
    password = fd.readline().strip('\n')
    """ Returns list of #### strings representing the file numbers that satisfy
    the given dict of constraints.
    ...
    Opens an SSH connection to the mitastrometry server, grabs all the Rb and
    .LOG files, shoves them into temp directories. For each extra constraint
    specified, looks in the fits header to check whether constraint is
    satisfied. If file is good, keep it in temp directory. If file is bad,
    delete it and its .LOG counterpart.
    """
    print(constraints)
    def progress(filename, size, sent):
        sys.stdout.write("%s\'s progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))
    rpath = 'R'
    if os.path.exists(rpath):
        while os.path.exists(rpath):
            rpath = rpath + '0'
    os.system('mkdir ' + rpath)
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('astrometry.mit.edu', username=username, password=password)
    with SCPClient(ssh.get_transport(), sanitize=lambda x: x, progress=progress) as scp:
        scp.get('/ast2/data/' + constraints['telescope'] + '/' + constraints['date'][:4] + '/' + constraints['date'].split('/')[0] +
                '/' + constraints['date'].split('/')[0] + '.R.*.gz')
    ssh.close()
    for file in glob.glob(constraints['date'].split('/')[0] + '.R.*.gz'):
        shutil.move(file, rpath + '/')
    filenames = os.listdir(rpath + '/')
    score = 0
    validated = []
    for f in filenames:
        if f[0] == '.':
            continue
        os.system("gunzip '" + rpath + '/' + f + "'")
        for key in constraints.keys():
            if key == 'instrument':
                value = fits.open(rpath + '/' + f[:-3])[0].header['INSTRUME']
                if value != constraints[key]:
                    score = score + 1
            elif key == 'object':
                value = fits.open(rpath + '/' + f[:-3])[0].header['OBJECT']
                if value != constraints[key]:
                    score = score + 1
        if score == 0:
            validated.append(f[-7:-3])
        else:
            print('removing ' + f)
            os.system("rm -rf " + rpath + "/" + f[:-3])
        score = 0
    return validated

def readnumbers():
    """Gets the numbers of the files that satisfied constraints and are sitting in the R/ file"""
    filenames = os.listdir(os.path.join(os.getcwd(),'R'))
    numbers= []
    for name in filenames:
        numbers.append(name[-4:])
    return numbers

def fetch(telescope, date, type, numbers=None): #type is R, Rb, or log
    try:
        fd = open(creds.txt)
    except FileNotFoundError:
        print('Create file creds.txt in this directory with pipeline username on line 1 and password on line 2')
    username = fd.readline().strip('\n')
    password = fd.readline().strip('\n')
    def progress(filename, size, sent):
        sys.stdout.write("%s\'s progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('astrometry.mit.edu', username=username, password=password)
    with SCPClient(ssh.get_transport(), sanitize=lambda x: x, progress=progress) as s:
        if type == 'R':
            if numbers is None:
                s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date.split('/')[0] +
                        '/' + date.split('/')[0] + '.R.*.gz')
            else:
                for number in numbers:
                    s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date.split('/')[0] +
                            '/' + date.split('/')[0] + '.R.' + number + '.gz')
        elif type == 'Rb':
            if numbers is None:
                s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date +
                    '/' + date.split('/')[0] + '.Rb.*')
            else:
                for number in numbers:
                    try:
                        s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date +
                                '/' + date.split('/')[0] + '.Rb.'+ number)
                    except scp.SCPException:
                        print("There is no Rb file for " + str(number))
                        os.system('rm -rf R/' + date + '.Rb.' + number)

        elif type == 'log':
            if numbers is None:
                s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date +
                    '/' + date.split('/')[0] + '.Rb.*.LOG')
            else:
                for number in numbers:
                    s.get('/ast2/data/' + telescope + '/' + date[:4] + '/' + date +
                            '/' + date.split('/')[0] + '.Rb.' + number + '.LOG')
        else:
            raise Exception('type must be R, Rb, or log')
    ssh.close()
    if type == 'R':
        if os.path.exists(type):
            while os.path.exists(type):
                type = type + '0'
        os.system('mkdir ' + type)
        for file in glob.glob(date.split('/')[0] + '.R.*.gz'):
            shutil.move(file, type +'/')
    elif type == 'Rb':
        if os.path.exists(type):
            while os.path.exists(type):
                type = type + '0'
        os.system('mkdir ' + type)
        for file in glob.glob(date.split('/')[0] + '.Rb.*'):
            shutil.move(file, type + '/')
    elif type == 'log':
        if os.path.exists(type):
            while os.path.exists(type):
                type = type + '0'
        os.system('mkdir ' + type)
        for file in glob.glob(date.split('/')[0] + '.Rb.*.LOG'):
            shutil.move(file, type + '/')
    else:
        raise Exception('type must be R, Rb, or log')
    print(str(len(os.listdir(type + '/'))) + ' ' + type + ' files successfully copied to ' + str(os.path.join(os.getcwd(), type)))


def run_cmd(sshClient, command):
    channel = sshClient.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(command)
    out = channel.makefile().read()
    err = channel.makefile_stderr().read()
    returncode = channel.recv_exit_status()
    channel.close()                       # channel is closed, but not the client
    return out, err, returncode


def readcsv(filepath, parse_nulls=True, fdp = False):
    """Reads an Rb file into a pandas dataframe"""
    data = {}
    keys = []
    fdp_data = {'parameter1':[], 'uncertainty1':[],
                'parameter2':[],'uncertainty2':[],'Fit1':[],'Fit2':[]}
    fdp_keys = ['parameter1', 'uncertainty1','parameter2',
                'uncertainty2', 'Fit1','Fit2']
    fdp_chi2 = []
    csv = open(filepath)
    keyindex = 0
    startindex = 0
    endindex = 1
    substring = ""
    linenumber = 0
    csv.readline()
    lines = csv.readlines()
    startline = len(lines)
    i = 0
    while i < len(lines):
        if lines[i][:9] == '#MATCHTOL' or i > startline:
            if lines[i][:9] == '#MATCHTOL':
                startline = i
                i = i + 2
            if lines[i][:19] == '#PlateFitChiSquared':
                startindex = 14
                endindex = startindex + 1
                keyindex = 0
                while endindex <= len(lines[i]):
                    substring = lines[i][startindex:endindex]
                    if substring[-1] == "." or (substring[0] == "-" and len(substring) == 1):
                        endindex = endindex + 1
                        substring = lines[i][startindex:endindex]
                    try:
                        number = float(substring)
                        if substring[-1] == " " or substring[-1] == "\t" or substring[-1] == "\n" or endindex == len(
                                lines):
                            fdp_chi2.append(number)
                            keyindex = keyindex + 1
                            startindex = endindex
                        endindex = endindex + 1
                    except ValueError:
                        startindex = endindex
                        endindex = endindex + 1
            if len(fdp_data['parameter1']) == 14:
                break
            startindex = 9
            endindex = startindex+1
            keyindex = 0
            while endindex <= len(lines[i]):
                substring = lines[i][startindex:endindex]
                if substring[-1] == "." or (substring[0] == "-" and len(substring) == 1) or substring[-1] == 'e':
                    endindex = endindex + 1
                    if substring[-1] == 'e':
                        endindex = endindex + 1
                    substring = lines[i][startindex:endindex]
                    #if linenumber == 46: #DEBUG
                        #print("line:", linenumber, " start:", startindex, " end:", endindex, ' substring:', substring, ' keyindex:', keyindex) #DEBUG
                try:
                    number = float(substring)
                    if substring[-1] == " " or substring[-1] == "\t" or substring[-1] == "\n" or endindex == len(lines):
                        #if linenumber == 46: #DEBUG
                        #print("appending " + substring + " to " + fdp_keys[keyindex]) #DEBUG
                        fdp_data[fdp_keys[keyindex]].append(number)
                        keyindex = keyindex + 1
                        startindex = endindex
                    endindex = endindex + 1
                except ValueError:
                    startindex = endindex
                    endindex = endindex + 1
        i = i+1
    csv.close()
    ###############################################################################
    csv = open(filepath)
    for line in csv:
        linenumber = linenumber + 1
        if line[:3] == '#1 ':
            startindex = 3
            endindex = 4
            while endindex <= len(line):
                substring = line[startindex:endindex]
                if substring[-1] == " " or substring[-1] == '\t' or endindex == len(line):
                    if substring.count(' ') != len(substring):
                        substring = substring.strip(' ')
                        substring = substring.strip('\n')
                        keys.append(substring)
                        startindex= endindex
                endindex= endindex + 1
            for key in keys:
                data[key] = []

        if line[0] == "#" or line [:2] == '\n':
            continue
        startindex = 0
        endindex = 1
        keyindex = 0
        while endindex <= len(line):
            substring = line[startindex:endindex]
            if substring[-1] == "." or (substring[0] == "-" and len(substring) == 1):
                endindex = endindex + 1
                substring = line[startindex:endindex]
            #if linenumber == 234: #DEBUG
                #print("line:", linenumber, " start:", startindex, " end:", endindex, ' substring:', substring, ' keyindex:', keyindex) #DEBUG
            try:
                number = float(substring)
                if substring[-1] == " " or substring[-1] == "\t" or substring[-1] == "\n" or endindex == len(line):
                    #if linenumber == 234: #DEBUG
                        #print("appending " + substring + " to " + keys[keyindex]) #DEBUG
                    data[keys[keyindex]].append(number)
                    keyindex = keyindex + 1
                    startindex = endindex
                endindex = endindex + 1
            except ValueError:
                startindex = endindex
                endindex = endindex + 1

    if parse_nulls:
        data = pd.DataFrame(data)
        data = data[[int(x) > 0 for x in data['RefID']]]
        data = data.reset_index()
    if not fdp:
        return data
    else:
        return data,(fdp_data, fdp_chi2)

#DEBUG READCSV#
#data, fdp = readcsv('mitastrometry/SARAS/20160611/20160611.Rb.150',fdp=True)
#data = pd.DataFrame(data)
#print(data, fdp)

def fetchlmi(date, object, filetype):
    """
    wrapper function for fetch() and satisfies_constraints() on DCT/LMI data
    scp's the data you want and deletes what you don't want.
    works by taking all the R files from the date dir, unzipping them,
    putting them in an R directory, and using the fits headers to delete
    everything that doesn't satisfy object constraint.
    Passes the numbers that satisfy object constraint to fetch(), which gets only
    the Rb/log files you want and puts them in an Rb/log folder.
    Run this function in the directory you want the R and Rb/log files of your
    target object in.
    It would be more efficient to use grep to find the files that satisfy
    constraints, but for reasons I can't recall I used this method instead
    ARGUMENTS
    ---------
    date (str) : Target date eg. '20190901a'
    object (str): Target object eg. '29P'
    filetype (str):'Rb' or 'log'
    """
    telescope = 'DCT/LMI'
    fetch(telescope, date + '/linearFDP2x2_GDR2', filetype,
        numbers = satisfies_constraints({'telescope': telescope,
                                        'date':date,
                                        'object':object}) )

def setuplmi(rpath, rbpath,
            centroid=False, make_output=False, use_preset_path=False):
    if use_preset_path:
        date = rpath
        number= rbpath
        data = readcsv('29p-data/'+date+'/Rb/'+date+'.Rb.'+number)
        fits_file = '29p-data/'+date+'/R/'+date+'.R.'+number
    else:
        date = rpath.split('/')[-1].split('.')[0]
        number = rpath.split('/')[-1].split('.')[2]
        data = readcsv(rbpath)
        fits_file = rpath

    image_data = fits.getdata(fits_file)
    if centroid:
        centroids = pd.read_csv('29p-data/29P_Mma_Centers.tsv', sep='\t')
        bools = [date in str(centroids['file'].iloc[i]) and number in str(centroids['file'].iloc[i]) for i in range(len(centroids))]
        centroid = centroids[bools]
        try:
            centroid = [centroid['X'].values[0], centroid['Y'].values[0]]
        except KeyError:
            print(number + 'not in centroid file')

    if make_output:
        starfield = StarField(telescope='DCT/LMI',date=date,
                                number=number,data = data, fits_file=fits_file)
        starfield.write_outfile()
    if centroid:
        return data, image_data, centroid
    else:
        return data, image_data

def findstar(rpath,rbpath, destdir='29p-data/', designation='centroid-offsets',save=True,
            size=100, subfield = None, lims = None, use_preset_path=False,nstars = 3,transform=True):
    #nstars is number of times you want it to loop
    #use_preset_path allows entry of date, number instead of rpath, rbpath with
    #the caveat that the file structure must be destdir/date/R/Rfiles for R and Rb
    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return idx

    if not use_preset_path:
        date = rpath.split('/')[-1].split('.')[0]
        number = rpath.split('/')[-1].split('.')[2]
    else:
        date=rpath
        number=rbpath
        rpath = os.path.join(destdir,date+'/R/'+date+'.R.'+number)
        rbpath = os.path.join(destdir,date+'/Rb/'+date+'.Rb.'+number)
    data = readcsv(rbpath)
    fits_file = rpath
    if subfield is None:
        image_data = fits.getdata(fits_file)
    else:
        image_data = subfield
        data['XIM'] = data['XIM']-lims[0][0]
        data['YIM'] = data['YIM']-lims[1][1]
    global count, indices
    count, indices = [], []
    def onclick(event):
        global ix, iy
        #global count
        ix, iy = event.xdata, event.ydata
        print('x = %d, y = %d'%(ix, iy))
        #Find closest entry in .Rb file to user click by
        #looking for where distance is closest to 0
        distances = np.sqrt((data['XIM'].values-ix)**2 +(data['YIM'].values-iy)**2)
        closest = find_nearest(distances, 0)
        indices.append(closest)
        centroid = [data['XIM'].iloc[closest], data['YIM'].iloc[closest]]
        distance=distances[closest]
        print('nearest star at:' + str(centroid))
        #plt.clf()
        plt.plot([ix,centroid[0]],[iy,centroid[1]],c='r'); #inform matplotlib of the new data
        plt.draw()
        count.append(0)
        if len(count) == nstars:
            fig.canvas.mpl_disconnect(cid)
            plt.close()

    #adjust image scale
    norm = ImageNormalize(image_data, interval=ZScaleInterval(),
                      stretch=LinearStretch())

    #Allow user to pick the centroid of their object
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_data, vmin = image_data.mean(),
               vmax = 2*image_data.mean(),cmap='viridis')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if save:
        if not os.path.exists(str(os.path.join(destdir,date))):
            os.system('mkdir ' +str(os.path.join(destdir,date)))
            os.system('mkdir ' +str(os.path.join(destdir,date + '/'+designation)))
        elif not os.path.exists(str(os.path.join(destdir,date+'/'+designation))):
             os.system('mkdir ' +str(os.path.join(destdir,date + '/'+designation)))
        plt.savefig(str(os.path.join(destdir,date+'/'+designation+'/'+'offsets'
                    + '_' + number+'.png')))

    return indices

def subfield_select(image_data):
    def on_xlims_change(event_ax):
        global xlim
        xlim = event_ax.get_xlim()
        #print("updated xlims: ", event_ax.get_xlim())

    def on_ylims_change(event_ax):
        global ylim
        ylim = event_ax.get_ylim()
        #print("updated ylims: ", event_ax.get_ylim())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    norm = ImageNormalize(image_data, interval=ZScaleInterval(),
                      stretch=LinearStretch())
    #ax.imshow(image_data, cmap = 'viridis',norm=norm)
    ax.imshow(image_data,vmin = image_data.mean(),
               vmax = 2*image_data.mean(),cmap='viridis')
    #ax.imshow(image_data, cmap = 'viridis')
    ax.callbacks.connect('xlim_changed', on_xlims_change)
    ax.callbacks.connect('ylim_changed', on_ylims_change)
    print('Close plot when selected window is satisfactory')
    plt.show()
    Xlim = [xlim[0],xlim[1]]
    Ylim = [ylim[0],ylim[1]]
    """if abs(int(Xlim[0])-int(Xlim[1])) % 2 != 0:
        Xlim[1] = Xlim[1]+1
    if abs(int(Ylim[0])-int(Ylim[1])) % 2 != 0:
        Ylim[1] = Ylim[1]+1"""
    print('X length:',str(abs(Xlim[0]-Xlim[1])))
    print('Y length:',str(abs(Ylim[0]-Ylim[1])))
    #Debug--
    image_data = image_data[int(Ylim[1]):int(Ylim[0]),int(Xlim[0]):int(Xlim[1])]
    norm = ImageNormalize(image_data, interval=ZScaleInterval(),
                      stretch=LinearStretch())
    print(image_data.shape)
    return image_data,Xlim,Ylim

#eg. fetchlmi('20191021b', '29P', 'Rb')
#eg. data, fits_file = setuplmi('path/to/date.R.number','path/to/date.Rb.number')
#           make_output=True if you want to instantiate StarField obj
