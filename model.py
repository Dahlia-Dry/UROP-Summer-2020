from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from utilities import *
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval, LinearStretch,ImageNormalize
import os
from itertools import compress
from mpl_toolkits.mplot3d import Axes3D
from centroid import Centroid
from scipy.optimize import curve_fit
import matplotlib.patches as patches
import time
from datetime import datetime


class CometPSF(object):
    def __init__(self,rdir,rbdir,outfile,fdp,eph,subset=None,core=False):
        """python wrapper for mathematica comet point spread function model and
           mathematica RA,Dec plate fit, with additional plotting functionality
           Parameters:
           -------------------------------------------------------------------
           rdir: string, path to directory containing R files
           rbdir: string, path to directory containing corresponding Rb files
           outfile: string, path to directory for output files to be written to
           fdp: string, path to relevant field distortion pattern grid file
           eph: string, path to ephemeris containing all observations in directory
                should be topocentric, formatted as csv with cols Time, RA, Dec
           subset: list of strings, default None, list of frames within R/Rb dirs to use
           core: boolean, default False, if True use just core fit instead of full fit with 1/r skirt
            -------------------------------------------------------------------"""
        self.rdir = rdir
        self.rbdir = rbdir
        self.outfile = outfile
        self.fdp = fdp
        self.eph = eph
        self.kernel = '/Applications/Mathematica 12.0.app/Contents/MacOS/WolframKernel' #location for pal@hubble.mit.edu
        self.subset = subset
        self.core = core

    def sort_frames(self):
        """Match each R file to its corresponding Rb file and align in a pandas dataframe
            output: Dataframe"""
        rdirlist = [x for x in os.listdir(self.rdir) if x[0] != '.']
        rbdirlist = [x for x in os.listdir(self.rbdir) if x[0] != '.']
        if len(rdirlist) != len(rbdirlist):
            raise Exception('Number of files in R and Rb directories must be equal')
        for i in range(len(rdirlist)):
            if not rdirlist[i].endswith('.fits'):
                os.rename(os.path.join(self.rdir,rdirlist[i]),os.path.join(self.rdir,rdirlist[i])+'.fits')
        rdirlist = [x for x in os.listdir(self.rdir) if x[0] != '.']
        df = pd.DataFrame({"frame":['' for x in rdirlist],
                            "r":[self.rdir for x in rdirlist],
                           "rb":[self.rbdir for x in rbdirlist],
                           "x0":[0 for x in rdirlist],
                           "y0":[0 for x in rdirlist],
                           "x0_adj":[0 for x in rdirlist],
                           "y0_adj":[0 for x in rdirlist],
                           "ra":[0 for x in rdirlist],
                           "dec":[0 for x in rdirlist],
                           "ra_offset":[0 for x in rdirlist],
                           "dec_offset":[0 for x in rdirlist]})
        for i in range(len(rdirlist)):
            print(rdirlist[i])
            rfile = rdirlist[i]
            rframe = rdirlist[i].split('.')[-2][-4:]
            for j in range(len(rbdirlist)):
                if rbdirlist[j][-4:] == rframe:
                    rbfile = rbdirlist[j]
            df['r'].iloc[i] = os.path.join(df['r'].iloc[i],rfile)
            df['rb'].iloc[i] = os.path.join(df['rb'].iloc[i],rbfile)
            df['frame'].iloc[i]= rframe
        print(df.head())
        df =df.set_index('frame')
        return df

    def get_coords(self,rframe, rbframe):
        """Displays image for user to select coordinates of 3 calibration stars and object
            Follow command line prompts to make selections
           writes input.csv for given frame to corresponding subdir in output folder
           output: [3x2 float] array of starcoords, [1x2 float] array of objectcoords """
        rpath = rframe
        rbpath = rbframe
        data = readcsv(rbpath)
        fits_file = rpath
        image_data = fits.getdata(fits_file)
        #step 1: choose subfield
        print('Select a window that includes the object and at least 3 calibration stars:')
        image_data,Xlim,Ylim = subfield_select(image_data)
        #step 2: pick stars to calculate sigma with
        print('Pick out 3 calibration stars;\
                window will close automatically once selections are made')
        rbindices = findstar(rpath=rpath,rbpath = rbpath,
                            nstars = 3, subfield = image_data,lims=[Xlim,Ylim], save=False)
        #step 3: isolate obj in its own subfield
        print('Use zoom to select a subfield containing only the object of interest;\
                close window when done')
        obj_data,objX, objY = subfield_select(image_data)
        fig = plt.figure()
        #Subplot 1 of 4: show apertures of fit windows
        ax = fig.add_subplot(111)
        ax.imshow(image_data, vmin = image_data.mean(),
                   vmax = 2*image_data.mean(),cmap='viridis')
        starcoords = []
        starap = 15
        for index in rbindices:
            #print('rb:',data.iloc[index]['XIM'],data.iloc[index]['YIM'])
            coords = (data.iloc[index]['YIM'],data.iloc[index]['XIM'])
            starcoords.append(coords)
        imgx = np.linspace(1,len(image_data[0]),len(image_data[0]))
        imgy = np.linspace(1,len(image_data),len(image_data))
        imgx, imgy = np.meshgrid(imgx,imgy)
        imgx = imgx.flatten()
        imgy = imgy.flatten()
        imgz = image_data.flatten()
        index = np.where(imgz == np.amax(obj_data))[0][0]
        objcoords = [int(imgy[index])+Ylim[1],int(imgx[index])+Xlim[0]]
        print('stars:',starcoords,'obj:',objcoords)
        return starcoords, objcoords

    def run_mathematicafit(self,frame,coords):
        """"Wraps PSF_fitting_v7.nb
            Uses mathematica to fit circular gaussian PSF with 1/r skirt to object in specified frame
            output: float x, float y centroid"""
        session = WolframLanguageSession(kernel=self.kernel)
        session.evaluate(wlexpr('<< jleGroup`'))
        session.evaluate(wlexpr('FileType[dir ="'+ self.rdir + '"]'))
        print('Directory:',session.evaluate(wlexpr('FileType[dir ="'+ self.rdir + '"]')))
        session.evaluate(wlexpr('filelist = FileNames["*.'+str(frame)+'.fits", dir]'))
        print('Files:',session.evaluate(wlexpr('filelist')))
        session.evaluate(wlexpr('frameStartTimesHeaders = faReadHeaderKeyword["DATE-OBS", #] & /@ filelist \n \
                                expTimesHeaders = ToExpression[faReadHeaderKeyword["EXPTIME", #]] & /@ filelist \n \
                                starTimesDateTimeStringHeaders = StringReplace[#, {"-" -> " ", "T" -> " "}] & /@ frameStartTimesHeaders \n \
                                obsStartTimesMJDHeaders = tcDateTimeStringToMJD[#] & /@ starTimesDateTimeStringHeaders \n \
                                expTimesDaysHeaders = ToExpression[ #]/60./60./24. & /@ expTimesHeaders; \n \
                                obsMidTimesMJDHeaders = obsStartTimesMJDHeaders + expTimesDaysHeaders/2. \n \
                                tcMJDtoDateTimeString[obsStartTimesMJDHeaders[[1]]] \n \
                                tcMJDtoDateTimeString[obsMidTimesMJDHeaders[[1]]]'))
        print(session.evaluate(wlexpr('frameStartTimesHeaders')))
        print(session.evaluate(wlexpr('expTimesHeaders')))
        print(session.evaluate(wlexpr('obsMidTimesMJDHeaders')))
        #print(session.evaluate(wlexpr('Do[frame[i]=ipLoadFrame[filelist[[i]]], {i, Length[filelist]}]')))
        session.evaluate(wlexpr('saturationLimit=64000;'))
        session.evaluate(wlexpr('objectApproxRowCol = {'+str(coords.loc['Object']['x'])+','+str(coords.loc['Object']['y'])+'};'))
        session.evaluate(wlexpr('starAapproxRowCol = {'+str(coords.loc['StarA']['x'])+','+str(coords.loc['StarA']['y'])+'};'))
        session.evaluate(wlexpr('starBapproxRowCol = {'+str(coords.loc['StarB']['x'])+','+str(coords.loc['StarB']['y'])+'};'))
        session.evaluate(wlexpr('starCapproxRowCol = {'+str(coords.loc['StarC']['x'])+','+str(coords.loc['StarC']['y'])+'};'))
        print('StarA Coords:',session.evaluate(wlexpr('starAapproxRowCol')))
        print('StarB Coords:',session.evaluate(wlexpr('starBapproxRowCol')))
        print('StarC Coords:',session.evaluate(wlexpr('starCapproxRowCol')))
        print('Obj Coords:',session.evaluate(wlexpr('objectApproxRowCol')))
        subframes= session.evaluate_wrap(wlexpr("""
                                    boxSizeObject = 40;
                                    boxSizeFieldStars = 40;
                                    objectApproxBox = {objectApproxRowCol-{boxSizeObject/2,boxSizeObject/2},{boxSizeObject,boxSizeObject}}
                                    starAapproxBox = {starAapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    starBapproxBox = {starBapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    starCapproxBox = {starCapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    frame[1] = ipLoadFrame[filelist[[1]]]
                                    objectSubFrame=ipSubFrame[frame[1],objectApproxBox];
                                    margAnal1=N[ipMargAnal[objectSubFrame]];
                                    newCtrObject=Round[margAnal1[[1]][[1]]]
                                    maxObject = Max[objectSubFrame[[3]]]*1.
                                    starAsubFrame=ipSubFrame[frame[1],starAapproxBox];
                                    margAnal2=N[ipMargAnal[starAsubFrame]];
                                    newCtrStarA=Round[margAnal2[[1]][[1]]]
                                    maxStarA = Max[starAsubFrame[[3]]]*1.
                                    starBsubFrame=ipSubFrame[frame[1],starBapproxBox];
                                    margAnal3=N[ipMargAnal[starBsubFrame]];
                                    newCtrStarB=Round[margAnal3[[1]][[1]]]
                                    maxStarB = Max[starBsubFrame[[3]]]*1.
                                    starCsubFrame=ipSubFrame[frame[1],starCapproxBox];
                                    margAnal4=N[ipMargAnal[starCsubFrame]];
                                    newCtrStarC=Round[margAnal4[[1]][[1]]]
                                    maxStarC = Max[starCsubFrame[[3]]]*1."""))
        starafit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starAapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarA,newCtrStarA[[1]],newCtrStarA[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starAfitResults = lsFitResultsForExport
                                                    """))
        print(starafit.result)
        starbfit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starBapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarB,newCtrStarB[[1]],newCtrStarB[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starBfitResults = lsFitResultsForExport
                                                    """))
        print(starbfit.result)
        starcfit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starCapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarC,newCtrStarC[[1]],newCtrStarC[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starCfitResults = lsFitResultsForExport
                                                    """))
        print(starcfit.result)
        objcorefitsetup = session.evaluate_wrap(wlexpr("""
                                                    lsUseModel[psMultipleSourceRelative[psCircularGaussianStarMesh,1]]
                                                    lsNames
                                                    lsCoor =ipPixelList[objectApproxBox];
                                                    lsData = ipDataList[frame[1],lsCoor];
                                                    fieldStarsFittedBkg = (starAfitResults[[2]][[1]]+starBfitResults[[2]][[1]]+starCfitResults[[2]][[1]])/3
                                                    fieldStarsFittedDiam =(starAfitResults[[2]][[-2]]+starBfitResults[[2]][[-2]]+starBfitResults[[2]][[-2]])/3
                                                    fieldStarsFittedShapeIndex = (starAfitResults[[2]][[-1]]+starBfitResults[[2]][[-1]]+starCfitResults[[2]][[-1]])/3
                                                    cutoff = fieldStarsFittedDiam*4
                                                    lsParamHist={lsParams=
                                                    Join[{fieldStarsFittedBkg,maxObject,newCtrObject[[1]],newCtrObject[[2]],
                                                    fieldStarsFittedDiam,fieldStarsFittedShapeIndex}]}
                                                    """))
        print(objcorefitsetup.result)
        print('FieldStarsFittedBkg',session.evaluate(wlexpr('fieldStarsFittedBkg')))
        print('fieldStarsFittedDiam',session.evaluate(wlexpr('fieldStarsFittedDiam')))
        print('FieldStarsFittedShapeIndex',session.evaluate(wlexpr('fieldStarsFittedShapeIndex')))
        print('cutoff',session.evaluate(wlexpr('cutoff')))
        annulusfxns = session.evaluate_wrap(wlexpr("""
                                                        includeAnnulus[c_,p_]:=
                                                            Module[{r0,c0, cut,rows,cols},
                                                                cut = p[[3]];
                                                                r0=p[[1]];
                                                                c0 = p[[2]];
                                                                rows = Take[Flatten[c],{1,-1,2}];
                                                                cols = Take[Flatten[c],{2,-1,2}];
                                                                dist = (r0-rows)^2 + (c0-cols)^2;
                                                                f[d_]:=If[(d != 0) && (d<cut),1,0];
                                                                weight= f/@dist
                                                                ]
                                                        excludeAnnulus[c_,p_]:=
                                                            Module[{r0,c0, cut,rows,cols},
                                                                cut = p[[3]];
                                                                r0=p[[1]];
                                                                c0 = p[[2]];
                                                                rows = Take[Flatten[c],{1,-1,2}];
                                                                cols = Take[Flatten[c],{2,-1,2}];
                                                                dist = (r0-rows)^2 + (c0-cols)^2;
                                                                f[d_]:=If[(d != 0) && (d>cut),1,0];
                                                                weight= f/@dist
                                                                ]
                                                        """))
        objcorefit = session.evaluate_wrap(wlexpr("""
                                                    lsWeight = includeAnnulus[lsCoor,{newCtrObject[[1]],newCtrObject[[2]],cutoff}]
                                                    lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                    Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                    lsFitList = {1,2,3,4}
                                                    lsChangeFrac=1.0;
                                                    lsIterateToConvergence[20,0]
                                                    objectFitResults = lsFitResultsForExport
                                                    """))
        print(objcorefit.result)
        cometfxns = session.evaluate_wrap(wlexpr("""
                                                    f[d_,cut_,k_]:=If[(d != 0) && (d>cut),k/d,0];
                                                    cometSkirtModel[p_,c_]:=
                                                        Module[{peak,r0,c0, cut,rows,cols,dist,k},
                                                                peak=p[[1]];
                                                                cut = p[[4]];
                                                                r0=p[[2]];
                                                                c0 = p[[3]];
                                                                k = p[[5]];
                                                                rows = Take[Flatten[c],{1,-1,2}];
                                                                cols = Take[Flatten[c],{2,-1,2}];
                                                                dist = (r0-rows)^2 + (c0-cols)^2;
                                                                skirt = f[#,cut,k] & /@dist
                                                                ]
                                                    combo[p_,c_]:=
                                                        Module[{pGauss, pBkg,pSkirt},
                                                                pGauss=Pick[p,{1,1,1,1,1,0,0,0},1];
                                                                pBkg={p[[6]], p[[2]],p[[3]]};
                                                                pSkirt=Pick[p,{1,1,1,0,0,0,1,1},1];
                                                                result= psCircularGaussianStarMesh[pGauss,c]+cometSkirtModel[pSkirt,c]+psConstantBackground[pBkg,c]
                                                                ]
                                                    lsInitialNames[combo]= {"peakSignal","rowCenter","colCenter","diameter","shapeIndex","background", "cutoff","k"}
                                                    lsInitialStep[model_]:=Table[0.01,{Length[lsInitialNames[model]]}]
                                                    lsUseModel[combo]
                                                    """))
        objskirtfitsetup = session.evaluate_wrap(wlexpr("""
                                                    lsNames
                                                    lsCoor =ipPixelList[objectApproxBox];
                                                    lsData = ipDataList[frame[1],lsCoor];
                                                    bkg = objectFitResults[[2]][[1]]
                                                    peak = objectFitResults[[2]][[2]]
                                                    r0 = objectFitResults[[2]][[3]]
                                                    c0 = objectFitResults[[2]][[4]]
                                                    diam = objectFitResults[[2]][[5]]
                                                    shape = objectFitResults[[2]][[6]]
                                                    k = peak
                                                    """))
        print(objskirtfitsetup.result)
        print('CORE FIT [bkg,peak,r0,c0] RESULTS:')
        print('Initial bkg:',session.evaluate(wlexpr('bkg')))
        print('Initial peak:',session.evaluate(wlexpr('peak')))
        print('Initial r0:',session.evaluate(wlexpr('r0')))
        print('Initial c0:',session.evaluate(wlexpr('c0')))
        print('Initial diam:',session.evaluate(wlexpr('diam')))
        print('Initial shape:',session.evaluate(wlexpr('shape')))
        print('Initial k:',session.evaluate(wlexpr('k')))
        objskirtfit = session.evaluate_wrap(wlexpr("""
                                                    lsParamHist={lsParams=Join[{peak,r0,c0,diam,shape,bkg, cutoff,k}]}
                                                    lsWeight = excludeAnnulus[lsCoor,{newCtrObject[[1]],newCtrObject[[2]],cutoff}]
                                                    lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                    Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                    lsFitList = {6,8}
                                                    lsChangeFrac=1.0;
                                                    lsIterateToConvergence[20,0]
                                                    objectFitResults = lsFitResultsForExport
                                                    """))
        #print(objskirtfit.result)
        objskirtfit2setup = session.evaluate_wrap(wlexpr("""
                                                        lsNames
                                                        lsCoor =ipPixelList[objectApproxBox];
                                                        lsData = ipDataList[frame[1],lsCoor];
                                                        peak = objectFitResults[[2]][[1]]
                                                        r0 = objectFitResults[[2]][[2]]
                                                        c0 = objectFitResults[[2]][[3]]
                                                        diam= objectFitResults[[2]][[4]]
                                                        shape= objectFitResults[[2]][[5]]
                                                        bkg = objectFitResults[[2]][[6]]
                                                        cutoff = objectFitResults[[2]][[7]]
                                                        k = objectFitResults[[2]][[8]]
                                                        """))
        print('SKIRT FIT [bkg,k] RESULTS')
        print('New bkg:',session.evaluate(wlexpr('bkg')))
        print('New k:',session.evaluate(wlexpr('k')))
        print('peak:',session.evaluate(wlexpr('peak')))
        print('r0:',session.evaluate(wlexpr('r0')))
        print('c0:',session.evaluate(wlexpr('c0')))
        print('diam:',session.evaluate(wlexpr('diam')))
        print('shape:',session.evaluate(wlexpr('shape')))
        print('cutoff:',session.evaluate(wlexpr('cutoff')))
        objskirtfit2 = session.evaluate_wrap(wlexpr("""
                                                    lsParamHist={lsParams=Join[{peak,r0,c0,diam,shape,bkg, cutoff,k}]}
                                                    lsWeight = excludeAnnulus[lsCoor,{newCtrObject[[1]],newCtrObject[[2]],cutoff}]
                                                    lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                    Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                    lsFitList = {2,3,6,8}
                                                    lsChangeFrac=1.0;
                                                    lsIterateToConvergence[20,0]
                                                    objectFitResults = lsFitResultsForExport
                                                    """))
        writeresults = session.evaluate_wrap(wlexpr("""
                                                    model3d = Table[{Take[Flatten[lsCoor],{1,-1,2}][[i]], Take[Flatten[lsCoor],{2,-1,2}][[i]],lsModel[[i]]},{i,1,Length[lsModel]}];
                                                    data3d = Table[{Take[Flatten[lsCoor],{1,-1,2}][[i]], Take[Flatten[lsCoor],{2,-1,2}][[i]],lsData[[i]]},{i,1,Length[lsModel]}];
                                                    xsection=Table[lsModel[[20*40+i]],{i,0,40}];
                                                    date =StringSplit[StringSplit[filelist[[1]],"/"][[5]],"."][[1]]"""))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_resids.csv"],lsResid]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_model.csv"],model3d]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_xsection.csv"],xsection]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_data.csv"],data3d]'))
        x0 = session.evaluate(wlexpr('objectFitResults[[2]][[2]]'))
        y0 = session.evaluate(wlexpr('objectFitResults[[2]][[3]]'))
        session.terminate()
        return x0,y0

    def run_mathematicafit_core(self,frame,coords):
        """"Wraps PSF_fitting_v7.nb but just the core fit
            Uses mathematica to fit circular gaussian PSF to object in specified frame
            output: float x, float y centroid"""
        session = WolframLanguageSession(kernel=self.kernel)
        session.evaluate(wlexpr('<< jleGroup`'))
        session.evaluate(wlexpr('FileType[dir ="'+ self.rdir + '"]'))
        print('Directory:',session.evaluate(wlexpr('FileType[dir ="'+ self.rdir + '"]')))
        session.evaluate(wlexpr('filelist = FileNames["*.'+str(frame)+'.fits", dir]'))
        print('Files:',session.evaluate(wlexpr('filelist')))
        session.evaluate(wlexpr('frameStartTimesHeaders = faReadHeaderKeyword["DATE-OBS", #] & /@ filelist \n \
                                expTimesHeaders = ToExpression[faReadHeaderKeyword["EXPTIME", #]] & /@ filelist \n \
                                starTimesDateTimeStringHeaders = StringReplace[#, {"-" -> " ", "T" -> " "}] & /@ frameStartTimesHeaders \n \
                                obsStartTimesMJDHeaders = tcDateTimeStringToMJD[#] & /@ starTimesDateTimeStringHeaders \n \
                                expTimesDaysHeaders = ToExpression[ #]/60./60./24. & /@ expTimesHeaders; \n \
                                obsMidTimesMJDHeaders = obsStartTimesMJDHeaders + expTimesDaysHeaders/2. \n \
                                tcMJDtoDateTimeString[obsStartTimesMJDHeaders[[1]]] \n \
                                tcMJDtoDateTimeString[obsMidTimesMJDHeaders[[1]]]'))
        print(session.evaluate(wlexpr('frameStartTimesHeaders')))
        print(session.evaluate(wlexpr('expTimesHeaders')))
        print(session.evaluate(wlexpr('obsMidTimesMJDHeaders')))
        #print(session.evaluate(wlexpr('Do[frame[i]=ipLoadFrame[filelist[[i]]], {i, Length[filelist]}]')))
        session.evaluate(wlexpr('saturationLimit=64000;'))
        session.evaluate(wlexpr('objectApproxRowCol = {'+str(coords.loc['Object']['x'])+','+str(coords.loc['Object']['y'])+'};'))
        session.evaluate(wlexpr('starAapproxRowCol = {'+str(coords.loc['StarA']['x'])+','+str(coords.loc['StarA']['y'])+'};'))
        session.evaluate(wlexpr('starBapproxRowCol = {'+str(coords.loc['StarB']['x'])+','+str(coords.loc['StarB']['y'])+'};'))
        session.evaluate(wlexpr('starCapproxRowCol = {'+str(coords.loc['StarC']['x'])+','+str(coords.loc['StarC']['y'])+'};'))
        print('StarA Coords:',session.evaluate(wlexpr('starAapproxRowCol')))
        print('StarB Coords:',session.evaluate(wlexpr('starBapproxRowCol')))
        print('StarC Coords:',session.evaluate(wlexpr('starCapproxRowCol')))
        print('Obj Coords:',session.evaluate(wlexpr('objectApproxRowCol')))
        subframes= session.evaluate_wrap(wlexpr("""
                                    boxSizeObject = 40;
                                    boxSizeFieldStars = 40;
                                    objectApproxBox = {objectApproxRowCol-{boxSizeObject/2,boxSizeObject/2},{boxSizeObject,boxSizeObject}}
                                    starAapproxBox = {starAapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    starBapproxBox = {starBapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    starCapproxBox = {starCapproxRowCol-{boxSizeFieldStars/2,boxSizeFieldStars/2},{boxSizeFieldStars,boxSizeFieldStars}}
                                    frame[1] = ipLoadFrame[filelist[[1]]]
                                    objectSubFrame=ipSubFrame[frame[1],objectApproxBox];
                                    margAnal1=N[ipMargAnal[objectSubFrame]];
                                    newCtrObject=Round[margAnal1[[1]][[1]]]
                                    maxObject = Max[objectSubFrame[[3]]]*1.
                                    starAsubFrame=ipSubFrame[frame[1],starAapproxBox];
                                    margAnal2=N[ipMargAnal[starAsubFrame]];
                                    newCtrStarA=Round[margAnal2[[1]][[1]]]
                                    maxStarA = Max[starAsubFrame[[3]]]*1.
                                    starBsubFrame=ipSubFrame[frame[1],starBapproxBox];
                                    margAnal3=N[ipMargAnal[starBsubFrame]];
                                    newCtrStarB=Round[margAnal3[[1]][[1]]]
                                    maxStarB = Max[starBsubFrame[[3]]]*1.
                                    starCsubFrame=ipSubFrame[frame[1],starCapproxBox];
                                    margAnal4=N[ipMargAnal[starCsubFrame]];
                                    newCtrStarC=Round[margAnal4[[1]][[1]]]
                                    maxStarC = Max[starCsubFrame[[3]]]*1."""))
        starafit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starAapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarA,newCtrStarA[[1]],newCtrStarA[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starAfitResults = lsFitResultsForExport
                                                    """))
        print(starafit.result)
        starbfit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starBapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarB,newCtrStarB[[1]],newCtrStarB[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starBfitResults = lsFitResultsForExport
                                                    """))
        print(starbfit.result)
        starcfit = session.evaluate_wrap(wlexpr("""
                                                lsUseModel[psMultipleSourceRelative[psCircularGaussianStar,1]]
                                                lsNames
                                                lsWeight = 1;
                                                lsCoor =ipPixelList[starCapproxBox];
                                                lsData = ipDataList[frame[1],lsCoor];
                                                lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                lsParamHist={lsParams=Join[{400,maxStarC,newCtrStarC[[1]],newCtrStarC[[2]],4,1.5}]}
                                                Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                lsFitList = {1,2,3,4,5,6};
                                                lsChangeFrac=1.0;
                                                lsIterateToConvergence[20,0];
                                                starCfitResults = lsFitResultsForExport
                                                    """))
        print(starcfit.result)
        objcorefitsetup = session.evaluate_wrap(wlexpr("""
                                                    lsUseModel[psMultipleSourceRelative[psCircularGaussianStarMesh,1]]
                                                    lsNames
                                                    lsCoor =ipPixelList[objectApproxBox];
                                                    lsData = ipDataList[frame[1],lsCoor];
                                                    fieldStarsFittedBkg = (starAfitResults[[2]][[1]]+starBfitResults[[2]][[1]]+starCfitResults[[2]][[1]])/3
                                                    fieldStarsFittedDiam =(starAfitResults[[2]][[-2]]+starBfitResults[[2]][[-2]]+starBfitResults[[2]][[-2]])/3
                                                    fieldStarsFittedShapeIndex = (starAfitResults[[2]][[-1]]+starBfitResults[[2]][[-1]]+starCfitResults[[2]][[-1]])/3
                                                    cutoff = fieldStarsFittedDiam*4
                                                    lsParamHist={lsParams=
                                                    Join[{fieldStarsFittedBkg,maxObject,newCtrObject[[1]],newCtrObject[[2]],
                                                    fieldStarsFittedDiam,fieldStarsFittedShapeIndex}]}
                                                    """))
        print(objcorefitsetup.result)
        print('FieldStarsFittedBkg',session.evaluate(wlexpr('fieldStarsFittedBkg')))
        print('fieldStarsFittedDiam',session.evaluate(wlexpr('fieldStarsFittedDiam')))
        print('FieldStarsFittedShapeIndex',session.evaluate(wlexpr('fieldStarsFittedShapeIndex')))
        print('cutoff',session.evaluate(wlexpr('cutoff')))
        annulusfxns = session.evaluate_wrap(wlexpr("""
                                                        includeAnnulus[c_,p_]:=
                                                            Module[{r0,c0, cut,rows,cols},
                                                                cut = p[[3]];
                                                                r0=p[[1]];
                                                                c0 = p[[2]];
                                                                rows = Take[Flatten[c],{1,-1,2}];
                                                                cols = Take[Flatten[c],{2,-1,2}];
                                                                dist = (r0-rows)^2 + (c0-cols)^2;
                                                                f[d_]:=If[(d != 0) && (d<cut),1,0];
                                                                weight= f/@dist
                                                                ]
                                                        excludeAnnulus[c_,p_]:=
                                                            Module[{r0,c0, cut,rows,cols},
                                                                cut = p[[3]];
                                                                r0=p[[1]];
                                                                c0 = p[[2]];
                                                                rows = Take[Flatten[c],{1,-1,2}];
                                                                cols = Take[Flatten[c],{2,-1,2}];
                                                                dist = (r0-rows)^2 + (c0-cols)^2;
                                                                f[d_]:=If[(d != 0) && (d>cut),1,0];
                                                                weight= f/@dist
                                                                ]
                                                        """))
        objcorefit = session.evaluate_wrap(wlexpr("""
                                                    lsWeight = includeAnnulus[lsCoor,{newCtrObject[[1]],newCtrObject[[2]],cutoff}]
                                                    lsUsePoints = If[lsData[[#]] > saturationLimit,0,1]& /@Range[Length[lsData]];
                                                    Transpose[{Range[Length[lsNames]],lsNames,lsParams}]//TableForm
                                                    lsFitList = {1,2,3,4}
                                                    lsChangeFrac=1.0;
                                                    lsIterateToConvergence[20,0]
                                                    objectFitResults = lsFitResultsForExport
                                                    """))
        writeresults = session.evaluate_wrap(wlexpr("""
                                                    model3d = Table[{Take[Flatten[lsCoor],{1,-1,2}][[i]], Take[Flatten[lsCoor],{2,-1,2}][[i]],lsModel[[i]]},{i,1,Length[lsModel]}];
                                                    data3d = Table[{Take[Flatten[lsCoor],{1,-1,2}][[i]], Take[Flatten[lsCoor],{2,-1,2}][[i]],lsData[[i]]},{i,1,Length[lsModel]}];
                                                    xsection=Table[lsModel[[20*40+i]],{i,0,40}];
                                                    date =StringSplit[StringSplit[filelist[[1]],"/"][[5]],"."][[1]]"""))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_resids.csv"],lsResid]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_model.csv"],model3d]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_xsection.csv"],xsection]'))
        session.evaluate(wlexpr('Export[StringJoin["'+os.path.join(self.outfile,frame)+'/'+'","psf_data.csv"],data3d]'))
        x0 = session.evaluate(wlexpr('objectFitResults[[2]][[3]]'))
        y0 = session.evaluate(wlexpr('objectFitResults[[2]][[4]]'))
        session.terminate()
        return x0,y0

    def fixed_size_subset(self,a, x, y, size):
        """helper function for fdp interpolation"""
        o, r = np.divmod(size, 2)
        l = (x-(o+r-1)).clip(0)
        u = (y-(o+r-1)).clip(0)
        a_ = a[l: x+o+1, u:y+o+1]
        out = np.full((size, size), np.nan, dtype=a.dtype)
        out[:a_.shape[0], :a_.shape[1]] = a_
        return out

    def plot_results(self,frame):
        """saves matplotlib fig as plots.png to corresponding frame subdir
           4 subplots, clockwise from top left: image, residuals, contour plots, 3d mesh"""
        model= pd.read_csv(os.path.join(self.outfile,frame+'/psf_model.csv'),
                            names=['x','y','values'],header=None)
        data = pd.read_csv(os.path.join(self.outfile,frame+'/psf_data.csv'),
                            names=['x','y','values'],header=None)
        print(model.head())
        print(data.head())
        print(np.meshgrid(model['x'],model['y']))
        xs = np.asarray(model['x'])
        ys = np.asarray(model['y'])
        img = np.asarray(data['values'])
        expected = np.asarray(model['values'])
        resids= np.asarray(data['values']-model['values'])
        cols = np.unique(xs).shape[0]
        X = xs.reshape(-1, cols)
        Y = ys.reshape(-1, cols)
        Z = resids.reshape(-1, cols)
        img = img.reshape(-1,cols)
        expected = expected.reshape(-1,cols)
        print(Z)
        #Make residual plot
        fig=  plt.figure()
        ax = fig.add_subplot(222)
        ax.set_title('Residuals')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        im = ax.imshow(Z,cmap='viridis',vmin=-200,vmax=200)
        fig.colorbar(im,ax=ax)
        #plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        #Make image plot
        ax = fig.add_subplot(221)
        ax.set_title('Image')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        norm = ImageNormalize(img, interval=ZScaleInterval(),
                          stretch=LinearStretch())
        im = ax.imshow(img,cmap='viridis')
        fig.colorbar(im,ax=ax)
        #Make 3d mesh plot
        ax = fig.add_subplot(223,projection='3d')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axes.zaxis.set_ticks([])
        ax.set_title('Model Wireframe/Data Contour')
        ax.plot_wireframe(self.fixed_size_subset(X,int(len(X[0])/2),int(len(X)/2),17),
                            self.fixed_size_subset(Y,int(len(X[0])/2),int(len(X)/2),17),
                            self.fixed_size_subset(expected,int(len(X[0])/2),int(len(X)/2),17))
        ax.contour(self.fixed_size_subset(X,int(len(X[0])/2),int(len(X)/2),17),
                            self.fixed_size_subset(Y,int(len(X[0])/2),int(len(X)/2),17),
                            self.fixed_size_subset(img,int(len(X[0])/2),int(len(X)/2),17),
                            rstride=4, cstride=4, linewidth=0.1,antialiased=True,cmap='plasma')
        """ax.plot_wireframe(self.fixed_size_subset(X,int(len(X[0])/2),int(len(X)/2),15),
                            self.fixed_size_subset(Y,int(len(X[0])/2),int(len(X)/2),15),
                            self.fixed_size_subset(img,int(len(X[0])/2),int(len(X)/2),15),colors='orange')"""
        #Make contour plot
        ax = fig.add_subplot(224)
        ax.set_title('Model Contour/Data Contour')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.contour(X,Y,img, cmap='Blues')
        ax.scatter(X[0][0],Y[0][0],c='blue',label='data')
        ax.contour(X,Y,expected,cmap='Greens')
        ax.scatter(X[0][0],Y[0][0],c='green',label='model')
        ax.legend()
        plt.savefig(os.path.join(self.outfile,frame+'/plots.png'),dpi=1200)

    def interpolate_fdp(self,x0,y0):
        """Linear interpolation of user-specified fdp to get adjusted centroid values
        output: float x0_adj, float y0_adj"""
        data = []
        file = open(self.fdp)
        for line in file:
            if line[0] != "#":
                data.append([x for x in line.split(' ')])
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = float(data[i][j].replace('\n',''))
        data =np.array(data)
        xvals = np.sort(np.unique(data[:,0]))
        yvals = np.sort(np.unique(data[:,1]))
        x1,x2 = xvals[np.searchsorted(xvals,x0)-1],xvals[np.searchsorted(xvals,x0)]
        y1,y2 = yvals[np.searchsorted(yvals,y0)-1],yvals[np.searchsorted(yvals,y0)]
        print(x1,x2,y1,y2)
        qx11,qy11,qx12,qy12,qx21,qy21,qx22,qy22 = -1,-1,-1,-1,-1,-1,-1,-1
        for row in data:
            if row[0] == x1 and row[1] == y1: #q11
                qx11 =row[3]
                qy11 =row[6]
            elif row[0] == x1 and row[1] == y2: #q12
                qx12 = row[3]
                qy12 = row[6]
            elif row[0] == x2 and row[1] == y1: #q21
                qx21 = row[3]
                qy21 = row[6]
            elif row[0] == x2 and row[1] == y2: #q22
                qx22 = row[3]
                qy22 = row[6]
            else:
                pass
        q = [qx11,qy11,qx12,qy12,qx21,qy21,qx22,qy22]
        if -1 in q:
            raise IndexError('1 or more FDP values not found')
        a1 = [x2-x0,x0-x1]
        ax2 = [[qx11,qx12],[qx21,qx22]]
        ay2 = [[qy11,qy12],[qy21,qy22]]
        a3 = [[y2-y0],[y0-y1]]
        c = 1/((x2-x1)*(y2-y1))
        dx= c*np.matmul(np.matmul(a1,ax2),a3)[0]
        dy= c*np.matmul(np.matmul(a1,ay2),a3)[0]
        print('dx:',dx)
        print('dy:',dy)
        return x0+dx,y0+dy

    def run_mathematicaradec(self,rb,x0,y0):
        """Wraps sel_pipeline_xy_to_radec.nb
            Converts given x0,y0 centroid to corresponding ra,dec coordinate in radians
            output: float ra, float dec (radians)"""
        session = WolframLanguageSession(kernel=self.kernel)
        session.evaluate(wlexpr('<< jleGroup`'))
        session.evaluate(wlexpr('rbFiles = {"'+rb+'"}'))
        print('File:',session.evaluate(wlexpr('rbFiles')))
        session.evaluate(wlexpr('rowObject= {'+str(x0)+'}'))
        session.evaluate(wlexpr('colObject= {'+str(y0)+'}'))
        calc = session.evaluate_wrap(wlexpr("""
                                        rbData = Import[#]&/@rbFiles;
                                        rbData[[1]];
                                        expTimesHeaders = Flatten[Table[ToExpression[StringDrop[rbData[[i]][[5]],9]],{i, Length[rbData]}]]
                                        dateDayFractions = Table[StringJoin[StringReplace[StringDrop[rbData[[i]][[3]],7],"-"->" "]<>StringDrop[ToString[ToExpression[StringDrop[rbData[[i]][[4]],4]][[1]]/24],1]],{i, Length[rbData]}]
                                        startTimeMJDs = tcDateDayFractionStringToMJD[#]&/@dateDayFractions
                                        midTimes = Table[startTimeMJDs[[i]]+expTimesHeaders[[i]]/2/60/60/24.,{i, Length[startTimeMJDs]}]
                                        fitHeaderPosition=Flatten[Table[Position[rbData[[i]],{"#PlateFit   parameter1 uncertainty1   parameter2 uncertainty2 Fit1 Fit2"}],{i, Length[rbData]}]]
                                        p1 = Table[Union[#]&/@StringSplit[rbData[[i]][[fitHeaderPosition[[i]]+1]]," "],{i, Length[rbData]}]
                                        p2 = Table[Union[#]&/@StringSplit[rbData[[i]][[fitHeaderPosition[[i]]+2]]," "],{i, Length[rbData]}]
                                        p3 = Table[Union[#]&/@StringSplit[rbData[[i]][[fitHeaderPosition[[i]]+3]]," "],{i, Length[rbData]}]
                                        p1[[1]][[1]]
                                        xip1 = ToExpression[Table[p1[[i]][[1]][[6]],{i, Length[rbData]}]]
                                        etap1 = ToExpression[Table[p1[[i]][[1]][[7]],{i, Length[rbData]}]]
                                        xip2 = ToExpression[Table[p2[[i]][[1]][[5]],{i, Length[rbData]}]]
                                        etap2 = ToExpression[Table[p2[[i]][[1]][[3]],{i, Length[rbData]}]]
                                        xip3 = ToExpression[Table[p3[[i]][[1]][[3]],{i, Length[rbData]}]]
                                        etap3 = ToExpression[Table[p3[[i]][[1]][[4]],{i, Length[rbData]}]]
                                        raCtr = Table[Union[#]&/@StringSplit[rbData[[i]][[fitHeaderPosition[[i]]-5]]," "],{i, Length[rbData]}]
                                        decCtr = Table[Union[#]&/@StringSplit[rbData[[i]][[fitHeaderPosition[[i]]-4]]," "],{i, Length[rbData]}]
                                        fitCtrRA = ToExpression[Column[Flatten[raCtr,1],2]]Degree
                                        fitCtrDec = ToExpression[Column[Flatten[decCtr,1],2]]Degree
                                        xiFromXY = Table[(xip1[[i]]+xip2[[i]]*colObject[[i]]+xip3[[i]]*rowObject[[i]])/206265,{i, Length[rbData]}]
                                        etaFromXY =Table[( etap1[[i]]+etap2[[i]]*colObject[[i]]+etap3[[i]]*rowObject[[i]])/206265,{i, Length[rbData]}]
                                        raMinusRAref = Table[ArcTan[Cos[fitCtrDec[[i]]] - etaFromXY[[i]]*Sin[fitCtrDec[[i]]], xiFromXY[[i]]],{i, Length[rbData]}]
                                        dec = Table[ArcTan[Cos[fitCtrDec[[i]]] - etaFromXY[[i]]*Sin[fitCtrDec[[i]]],Cos[raMinusRAref[[i]]]*(Sin[fitCtrDec[[i]]] + etaFromXY[[i]]*Cos[fitCtrDec[[i]]])],{i, Length[rbData]}]
                                        finalRAdecs=Table[{tcRadiansToRAstring[(raMinusRAref[[i]] + fitCtrRA[[i]] )],tcRadiansToDecString[dec[[i]]]},{i, Length[rbData]}]
                                        Transpose[{tcHMSstringToRadians[#]&/@Column[finalRAdecs,1],tcDecStringToRadians[#]&/@Column[finalRAdecs,2]}]
                                        """))
        print(calc.result)
        ra = calc.result[0][0]
        dec = calc.result[0][1]
        session.terminate()
        return ra,dec

    def get_jploffsets_nointerpolate(self,rb,ra_obs,dec_obs):
        """Deprecated, initial version that used ephemeris csv directly from Horizons
           and did not interpolate or use image mid-time"""
        f = open(rb)
        for line in f:
            if line.startswith('#Date='):
                date = line.replace('\n','').split(' ')[-1]
            elif line.startswith('#UT='):
                time = float(line.replace('\n','').split(' ')[-1])
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])
        day = int(date.split('-')[2])
        hour = int(time)
        minute = int((time*60)%60)
        second = int((time*3600)%60)
        obstime = datetime(year=year,month=month,day=day,
                            hour=hour,minute=minute)
        obs_offset = datetime(year=year,month=month,day=day,
                            hour=hour,minute=(minute+1)%60)
        ephemeris = pd.read_csv(self.eph)
        ephemeris['Time'] = pd.to_datetime(ephemeris['Time'])
        ephemeris = ephemeris[(ephemeris['Time']>=obstime) & (ephemeris['Time']<=obs_offset)]
        try:
            ra0 = ephemeris['RA'].iloc[0].strip(' ')
            ra1 = ephemeris['RA'].iloc[1].strip(' ')
            dec0 = ephemeris['Dec'].iloc[0].strip(' ')
            dec1 = ephemeris['Dec'].iloc[1].strip(' ')
            ra0 = float(ra0.split(' ')[0])*360/24 +float(ra0.split(' ')[1])*360/(24*60) + float(ra0.split(' ')[2])*360/(24*3600)
            ra1 = float(ra1.split(' ')[0])*360/24 +float(ra1.split(' ')[1])*360/(24*60) + float(ra1.split(' ')[2])*360/(24*3600)
            dec0 = float(dec0.split(' ')[0])+float(dec0.split(' ')[1])/60 + float(dec0.split(' ')[2])/3600
            dec1 = float(dec1.split(' ')[0])+float(dec1.split(' ')[1])/60 + float(dec1.split(' ')[2])/3600
            ra_eph = (ra0+ra1)/2
            dec_eph = (dec0+dec1)/2
            ra_offset = np.radians(ra_obs-ra_eph)/np.cos(np.radians(dec_eph))
            dec_offset = np.radians(dec_obs-dec_eph)
        except:
            ra_offset,dec_offset= -1,-1
        print(ra_offset,dec_offset)
        return ra_offset,dec_offset

    def get_jploffsets(self,rb,ra_obs,dec_obs):
        """Wraps sel_pipeline_xy_to_radec.nb
           Uses user-provided .eph file to calculate ra and dec offsets from JPL (radians)
           output: float ra_offset, float dec_offset (radians)"""
        f = open(rb)
        for line in f:
            if line.startswith('#Date='):
                date = line.replace('\n','').split(' ')[-1]
            elif line.startswith('#UT='):
                time = float(line.replace('\n','').split(' ')[-1])
            elif line.startswith('#Exptime='):
                exptime = float(line.replace('\n','').split(' ')[-1])
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])
        day = int(date.split('-')[2])
        time = time + (exptime/2)/3600
        hour = int(time)
        minute = int((time*60)%60)
        second = int((time*3600)%60)
        session = WolframLanguageSession(kernel=self.kernel)
        session.evaluate(wlexpr('<< jleGroup`'))
        session.evaluate(wlexpr('eph = "'+self.eph+'"'))
        session.evaluate(wlexpr('Timing[ ({interpEphemRA, interpEphemDec} = etInterpolatingFunctionsForRAandDec[ "Horizons", "OBS", eph ] ); ]'))
        mjd = session.evaluate_wrap(wlexpr('midTimesFromHeaders = {tcDateTimeStringToMJD["'+str(year)+' '+str(month)+' '+str(day)+' '+str(hour)+':'+str(minute)+':'+str(second)+'"]}'))
        print('MJD:',mjd.result)
        jpl = session.evaluate_wrap(wlexpr('{interpEphemRA[#], interpEphemDec[#]}&/@{midTimesFromHeaders[[1]]}')).result
        session.terminate()
        #ra_obs =np.radians(ra_obs)
        #dec_obs = np.radians(dec_obs)
        ra_eph = jpl[0][0]
        dec_eph= jpl[0][1]
        print('RA,DEC:',ra_eph,dec_eph)
        ra_offset = (ra_obs-ra_eph)/np.cos(dec_eph)
        dec_offset = dec_obs-dec_eph
        return ra_offset, dec_offset

    def fit_imgs(self):
        """Uses the above functions to fit the data within the user provided directories"""
        df = self.sort_frames()
        if self.subset is not None:
            df =df.loc[self.subset]
        #1. User Input to determine coordinates of calibration stars and
        #   object subfield in each frame
        for i in range(len(df)):
            frame= df['r'].iloc[i].split('.')[-2][-4:]
            if not os.path.exists(os.path.join(self.outfile,df['r'].iloc[i].split('.')[-2][-4:]+'/input.csv')):
                print(frame)
                starcoords, objectcoords = self.get_coords(df['r'].iloc[i],df['rb'].iloc[i])
                coords = [c for c in starcoords]
                coords.append(objectcoords)
                inputcoords = pd.DataFrame({'label':['StarA','StarB','StarC','Object'],
                                            'x':[int(c[0]) for c in coords],
                                            'y':[int(c[1]) for c in coords],}).set_index('label')
                if not os.path.exists(os.path.join(self.outfile,df['r'].iloc[i].split('.')[-2][-4:])):
                    os.system('mkdir '+os.path.join(self.outfile,df['r'].iloc[i].split('.')[-2][-4:]))
                inputcoords.to_csv(os.path.join(self.outfile,df['r'].iloc[i].split('.')[-2][-4:]+'/input.csv'))
            else:
                inputcoords= pd.read_csv(os.path.join(self.outfile,df['r'].iloc[i].split('.')[-2][-4:]+'/input.csv')).set_index('label')
        print('Fitting x,y')
        #2. Fit centroids for each frame in mathematica
        for i in range(len(df)):
            frame= df['r'].iloc[i].split('.')[-2][-4:]
            try:
                if self.core:
                    x0,y0 = self.run_mathematicafit_core(frame,inputcoords)
                else:
                    x0,y0 = self.run_mathematicafit(frame,inputcoords)
            except:
                x0,y0=-1,-1
                time.sleep(10)
            df['x0'].iloc[i] = x0
            df['y0'].iloc[i] = y0
            self.plot_results(frame)
        df.to_csv(os.path.join(self.outfile,'coords.csv'))
        #3. Linearly Interpolate x0,y0 with FDP to get adjusted coords,
        #   then run mathematica plate fit to get RA, Dec
        df = pd.read_csv(os.path.join(self.outfile,'coords.csv')).set_index('frame')
        print('Fitting RA, Dec')
        for i in range(len(df)):
            rb = df['rb'].iloc[i]
            x0 = df['x0'].iloc[i]
            y0 = df['y0'].iloc[i]
            if x0 != -1 and y0 != -1:
                try:
                    x0_adj,y0_adj = self.interpolate_fdp(x0,y0)
                except:
                    x0_adj,y0_adj = -1,-1
                df['x0_adj'].iloc[i] = x0_adj
                df['y0_adj'].iloc[i] = y0_adj
                if x0_adj != -1 and y0_adj != -1:
                    ra, dec = self.run_mathematicaradec(rb,x0_adj,y0_adj)
                    df['ra'].iloc[i] = ra
                    df['dec'].iloc[i] = dec
                    ra_offset, dec_offset = self.get_jploffsets(rb,ra,dec)
                    df['ra_offset'].iloc[i] = ra_offset
                    df['dec_offset'].iloc[i] = dec_offset
        df.to_csv(os.path.join(self.outfile,'coords.csv'))

#example usage
rdir =os.path.abspath('29p-data/20190901a/R/')
rbdir =os.path.abspath('29p-data/20190901a/Rb/')
outfile = os.path.abspath('29p-results/20190901a_corefit/')
fdp = os.path.abspath('29p-data/FDP_Grid.3069x3076.32')
eph = os.path.abspath('29p-ephemeris.eph')
psf = CometPSF(rdir,rbdir,outfile,fdp,eph,core=True)
psf.fit_imgs()
