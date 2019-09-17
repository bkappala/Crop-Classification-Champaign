import numpy as np
from sklearn import cluster
from osgeo import gdal, gdal_array

gdal.UseExceptions()
gdal.AllRegister()

img_ds = gdal.Open('20130824_RE3_3A_Analytic_Champaign_north.tif', gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
print (img.shape)
print (new_shape)

X = img[:, :, :13].reshape(new_shape)
print (X.shape)


# read output variables
cdal_ds = gdal.Open('CDL_2013_Champaign_north.tif', gdal.GA_ReadOnly)

cdal_img = np.zeros((cdal_ds.RasterYSize, cdal_ds.RasterXSize, cdal_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(cdal_ds.GetRasterBand(1).DataType))

for b in range(cdal_img.shape[2]):
    cdal_img[:, :, b] = cdal_ds.GetRasterBand(b + 1).ReadAsArray()
    
cdal_shape = (cdal_img.shape[0] * cdal_img.shape[1], cdal_img.shape[2])

Y = cdal_img[:, :, :13].reshape(cdal_shape)
print (Y.shape)

#Change Y values to 3 values instead of 255 colors
Y[(Y != 1) & (Y != 5)] = 0
np.unique(Y, return_counts=True)


from sklearn.cross_validation import train_test_split

# Test purposes we will reduce the number from 56163575 to 10 percent to make faster 
seed = 7
test_size = 0.80
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print(X_train.shape)
print(y_train.shape)

# Now we have 10% of the data only and we use this to train and test
# NOTE: since it is unsupervised classification, we dont need to divide anymore
seed = 13
test_size = 0.23
X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)
print(X1_train.shape)
print(y1_train.shape)


# t1 = copy.deepcopy(Y)

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_jobs=6)

#rf.fit(X1_train, y1_train)
rf.fit(X, Y)

#predict = rf.predict(X1_test)
#from sklearn.metrics import accuracy_score

#accuracy = accuracy_score(predict, y1_test)

#print(accuracy)


print("Finished Model .. now test writing")

south_img_ds = gdal.Open('20130824_RE3_3A_Analytic_Champaign_south.tif', gdal.GA_ReadOnly)

south_img = np.zeros((south_img_ds.RasterYSize, south_img_ds.RasterXSize, south_img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(south_img_ds.GetRasterBand(1).DataType))

for b in range(south_img.shape[2]):
    south_img[:, :, b] = south_img_ds.GetRasterBand(b + 1).ReadAsArray()
    
south_new_shape = (south_img.shape[0] * south_img.shape[1], south_img.shape[2])
print (south_img.shape)
print (south_new_shape)

XS = south_img[:, :, :13].reshape(south_new_shape)
print (XS.shape)

predict = rf.predict(XS)


# SRINU change Y to correct
print(predict.shape)
#[cols, rows] = predict.shape
predict = predict.reshape(5959,9425)
[cols, rows] = predict.shape
format = "GTiff"
driver = gdal.GetDriverByName(format)

outDataRaster = driver.Create("test1.tif", rows, cols, 1, gdal.GDT_Byte)
outDataRaster.SetGeoTransform(cdal_ds.GetGeoTransform())##sets same geotransform as input
outDataRaster.SetProjection(cdal_ds.GetProjection())##sets same projection as input


outDataRaster.GetRasterBand(1).WriteArray(predict)

outDataRaster.FlushCache() ## remove from memory
del outDataRaster ## delete the data (not the actual geotiff)

print("Finished Writing")
