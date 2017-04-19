#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:53:57 2017

@author: guang
"""
import config as _config
import os
import pandas as pd
from pandas import DataFrame
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


_cur_dir = os.path.realpath(__file__)
_cur_dir = os.path.dirname(_cur_dir)

_OFFICIAL_ELLIPSE = True

with open(os.path.join(_cur_dir, 'annotations/folds.pkl')) as f:
    _folds = pk.load(f)

_annotations = pd.read_csv(os.path.join(_cur_dir, 'annotations/annotation1024.csv'))
_annotations[['x','y','x1','x2','x3','x4','y1','y2','y3','y4']] -= 1

# Precompute the axises of ellipse
_annotations['width'] = None
_annotations['length'] = None

_annotations['orient_degree'] = _annotations['orient'] * 180 / np.pi 



for i, row in _annotations.iterrows():
    
    sides = []
    if _OFFICIAL_ELLIPSE:
    # Original algorithm in official Devkit    
        sides.append([(row['x1']-row['x2'])+1, (row['y1']-row['y2'])+1])
        sides.append([(row['x2']-row['x3'])+1, (row['y2']-row['y3'])+1])
        sides.append([(row['x3']-row['x4'])+1, (row['y3']-row['y4'])+1])
        sides.append([(row['x4']-row['x1'])+1, (row['y4']-row['y1'])+1])
    else:
    # More reasonable
        sides.append([abs(row['x1']-row['x2'])+1, abs(row['y1']-row['y2'])+1])
        sides.append([abs(row['x2']-row['x3'])+1, abs(row['y2']-row['y3'])+1])
        sides.append([abs(row['x3']-row['x4'])+1, abs(row['y3']-row['y4'])+1])
        sides.append([abs(row['x4']-row['x1'])+1, abs(row['y4']-row['y1'])+1])    
        
        
    sides = np.array(sides)
    
    l = []
    for n in range(4):
        l.append(np.linalg.norm(sides[n]))
    l.sort()
    length = np.mean(l[2:4])
    width = np.mean(l[0:2])    
    _annotations.loc[i, 'width'] = width
    _annotations.loc[i, 'length'] = length

del i, l, n, row, sides, length, width

    
_legal_image_ids = set(_folds['train'][0]) | set(_folds['test'][0])


_vehicle_classes = {
    31:'plane',
    23:'boat',
    5:'camping car',
    1:'car',
    11:'pick-up',
    4:'tractor',
    2:'truck',
    9:'van',
    10:'other',
}

_vehicle_classes_ids = {    
    'car':1,
    'tractor':4,    
    'van':9,    
    'pick-up':11,
    'small land vehicle':set([1, 4, 9, 11]),

    'truck':2,
    'camping car':5,
    'large land vehicle':set([2, 5]),

    'other':10,
    'boat':23,
    'plane':31,
}

def get_image_path(image_id, image_type='co'):
    if not image_type in ['co', 'ir']:
        raise ValueError('{} is not a legal image type. Expect "co" or "ir".'.format(image_type))
    image_id = int(image_id)
    if not image_id in _legal_image_ids:
        raise ValueError('{} is not a legal image ID.'.format(image_id))

    image_name = '{:08d}_{}.png'.format(image_id, image_type)
    return str(os.path.join(_config.PATH_VEHICULES_1024, image_name))
    
    
IMG_SIZE = 1024    

def visualize_annotations(image_id, image_type='co', vehicle_marker='None', vehicle_shape='polygon'):
    image = Image.open(get_image_path(image_id, image_type))
    plt.imshow(image)
    ax = plt.gca()
    image_label = _annotations.loc[_annotations['image_id'] == image_id]
    for index, row in image_label.iterrows():
        
#        plt.scatter(row['x'], row['y'], 
#                    s=40, c='None', 
#                    edgecolor='#008800', 
#                    marker=vehicle_marker, 
#                    lw = 4, zorder=10)
#        plt.scatter(row['x'], row['y'], 
#                    s=40, c='None', 
#                    edgecolor='#00FF00', 
#                    marker=vehicle_marker, 
#                    lw = 2, zorder=10)
        vehicle_name = _vehicle_classes[row['class']]
        plt.plot([row['x'], row['x']], [row['y'], row['y']-40], color='black', lw=1, alpha=0.4)
        ax.text(row['x'], row['y'] - 40, vehicle_name, color='white', 
                horizontalalignment='center', fontsize=9, 
                bbox={'facecolor':'black', 'alpha':0.4}
                )
        
        if vehicle_shape == 'polygon':
            xs = [
                    row['x1'],
                    row['x2'],
                    row['x3'],
                    row['x4'],
                    row['x1'],
            ]
            ys = [
                    row['y1'],
                    row['y2'],
                    row['y3'],
                    row['y4'],
                    row['y1'],
            ]
            plt.plot(xs, ys, linewidth=1, color='#00FF00')
            #plt.plot(xs, ys, linewidth=1, color='#00FF00')            
        elif vehicle_shape == 'ellipse':
            patch = mpatches.Ellipse([row['x'], row['y']], 
                                        row['length'], 
                                        row['width'], 
                                        row['orient_degree'], 
                                        fc='none', 
                                        ls='-', 
                                        ec='#00FF00', 
                                        lw=1, 
                                        #zorder=10
                                        alpha=0.5
                                        )
            ax.add_patch(patch)
#            patch = mpatches.Ellipse([row['x'], row['y']], 
#                                        row['length'], 
#                                        row['width'], 
#                                        row['orient_degree'], 
#                                        fc='none', 
#                                        ls=':', 
#                                        ec='#FFFFFF', 
#                                        lw=3, 
#                                        #zorder=10
#                                        alpha=0.5
#                                        )
#            ax.add_patch(patch)        
        #vehicle_name = _vehicle_classes[row['class']] 
        #currentAxis.text(min(xs), min(ys)-5, vehicle_name, color='black', fontsize=12, bbox={'facecolor':'#FFFF00', 'alpha':0.7})
    plt.xlim([0, IMG_SIZE])
    plt.ylim([IMG_SIZE, 0])    
        
    
def visualize_prediction(image_id, prediction, thres = -10.0, image_type='co', vehicle_marker='None', collision_detection=False, verbose=False):    
    image = Image.open(get_image_path(image_id, image_type))
    plt.imshow(image)
    ax = plt.gca()    
    
    prediction_this_image = prediction.loc[(prediction['image_id']==image_id) & (prediction['score']>=thres)]

    if collision_detection:
        gt_this_image = _annotations.loc[_annotations['image_id']==image_id]        

    if verbose:
        print prediction_this_image
    for i, row in prediction_this_image.iterrows():
#        if row['score'] < thres: continue

        marker_fc = '#FFFF00'
        marker_ec = '#888800'
        
        if collision_detection:
            marker_fc = '#FF4444'
            marker_ec = '#880000'       
            for j, gt_instance in gt_this_image.iterrows():
                if collision_check(row, gt_instance):
                    marker_fc = '#44FF44'
                    marker_ec = '#008800'

             
    
        plt.scatter(row['x'], row['y'], 
                    s=20, c='None', 
                    edgecolor=marker_ec, 
                    marker=vehicle_marker, 
                    lw = 4, zorder=10)
        plt.scatter(row['x'], row['y'], 
                    s=20, c='None', 
                    edgecolor=marker_fc, 
                    marker=vehicle_marker, 
                    lw = 2, zorder=10)
        
        ax.text(row['x'], row['y'] - 10, '{:05.3f}'.format(row['score']), color='black', 
                horizontalalignment='center', fontsize=9, 
                bbox={'facecolor':marker_fc, 'alpha':0.4}
                )
    plt.xlim([0, IMG_SIZE])
    plt.ylim([IMG_SIZE, 0])           
    
    
def load_prediction_txt(file_path):
    prediction = DataFrame(columns=['image_id', 'x', 'y', 'score'])
    with open(file_path) as f:
        for line in f.readlines():
            image_id, x, y, score = line.split(' ')
            row_id = prediction.shape[0]
            prediction.loc[row_id, 'image_id'] = int(image_id)
            prediction.loc[row_id, 'x'] = float(x)    
            prediction.loc[row_id, 'y'] = float(y)      
            prediction.loc[row_id, 'score'] = float(score)     
    return prediction
    
    
def collision_check(point, ellipse):
    X = point['x'] - ellipse['x']
    Y = point['y'] - ellipse['y']
    alpha = - ellipse['orient']
    x = np.cos(alpha) * X - np.sin(alpha) * Y
    y = np.sin(alpha) * X + np.cos(alpha) * Y
    x /= ellipse['length'] * 0.5
    y /= ellipse['width'] * 0.5
    return np.square(x) + np.square(y) <= 1.0


def evaluate(prediction, vehicle_class, fold):
    cids = _vehicle_classes_ids[vehicle_class]
    if type(cids) == int:
        gt_this_class = _annotations.loc[(_annotations['class']==cids) & (_annotations['test_fold']==fold)]
    else:
        gt_this_class = _annotations.loc[_annotations['class'].isin(cids) & (_annotations['test_fold']==fold)]
    gt_this_class['recalled'] = False

    nb_test_samples = len(_folds['test'][fold])

    print 'Fold: {}. Total number of "{}" instances: {}'.format(fold, vehicle_class, gt_this_class.shape[0])                        
    
    prediction = prediction.sort_values(by='score', axis=0, ascending=False)
    
    positive_scores = []
    negative_scores = []

    
    for i, pred_instance in prediction.iterrows():
        image_id = pred_instance['image_id']
        gt_this_image = gt_this_class.loc[gt_this_class['image_id']==image_id]

        hit = False
        for j, gt_instance in gt_this_image.iterrows():
            if collision_check(pred_instance, gt_instance):
                hit = True
                if not gt_instance['recalled']:
                    gt_this_class.loc[j, 'recalled'] = True
                    positive_scores.append(pred_instance['score'])
                    break
        if not hit:
            negative_scores.append(pred_instance['score'])
    
    negative_scores = np.array(negative_scores)        
    eval_results = DataFrame(columns=['score', 'nb_pos', 'nb_neg', 'precision', 'recall', 'FPPI'])
    for i, score in enumerate(positive_scores):
        nb_pos = i+1
        nb_neg = np.sum(negative_scores >= score)
        eval_results.loc[i, 'score'] = score
        eval_results.loc[i, 'nb_pos'] = nb_pos
        eval_results.loc[i, 'nb_neg'] = nb_neg
        eval_results.loc[i, 'precision'] = float(nb_pos) / (nb_pos+nb_neg) 
        eval_results.loc[i, 'recall'] = float(nb_pos) / gt_this_class.shape[0]
        eval_results.loc[i, 'FPPI'] = float(nb_neg) / nb_test_samples

    return eval_results


def precision_recall_11points(eval_results):
    recalls = np.linspace(0.0, 1.0, 11)
    precisions = []
    for recall in recalls:
        prec_smooth = eval_results.loc[eval_results['recall']>=recall, 'precision'].max()
        if np.isnan(prec_smooth):
            prec_smooth = 0.0
        precisions.append(prec_smooth)
    recalls = list(recalls)
    
#    ap = 0.0
#    for i in range(10):
#        ap += 0.1 * 0.5 * (precisions[i]+precisions[i+1])
    ap = np.mean(precisions)        


    return precisions, recalls, ap
    
    
def _linear_interpolate(x, X, Y):
    segments = np.logical_and(x >= X[:-1], x <= X[1:])    
    segment = segments.tolist().index(True)    
    y = Y[segment]+(Y[segment+1] - Y[segment])*(x - X[segment])/(X[segment+1] - X[segment])
    return y
    
def recall_FPPI(eval_results):
    recalls = list(eval_results['recall'])
    FPPIs = list(eval_results['FPPI'])
        
    recalls.insert(0, 0)
    FPPIs.insert(0, 0) 
    
    recalls.append(recalls[-1])
    FPPIs.append(1e8)
    
    recalls = np.array(recalls)
    FPPIs = np.array(FPPIs)
    
    recalls_output = []
    fppis_output = []

    for FPPI in [0.001, 0.01, 0.1, 1.0, 10.0]:
        recalls_output.append(_linear_interpolate(FPPI, FPPIs, recalls))
        fppis_output.append(FPPI)
    return recalls_output, fppis_output
    
if __name__ == "__main__":
    image_id = 770
    visualize_annotations(image_id, vehicle_marker='None', vehicle_shape='None')
