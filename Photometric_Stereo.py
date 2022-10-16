import numpy as np
import iamge_operations
import cv2
import scipy
import scipy.misc, scipy.sparse
import scipy.sparse.linalg
from PIL import Image
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix

class photometric_stereo():
    def __init__(self,images,mask_image,light_inform):
        self.images=images                       #图像
        self.mask_image=mask_image               #掩模
        self.light_inform=light_inform           #照射光信息-光源方向向量
        h,w=mask_image.shape
        self.normal_vectors_matrix=np.zeros((h,w,3))                   #法向量矩阵
        self.depth_matrix=np.zeros((h,w,3))                            #深度矩阵
        
        
    def normal_vectors_computation(self):#计算法向量信息
        self.normal_vectors_matrix[:, :, 2] = 1
        '''I=[]#存放图片拉成行向量后的像素信息矩阵I
        for image in self.images:
            image=image.reshape((-1,1)).squeeze(1)#将图片拉成行向量
            I.append(image)
        I=np.array(I)

        N=np.linalg.lstsq(self.light_inform, I, rcond=-1)[0].T
        N=normalize(N,axis=1)#归一化
        self.normal_vectors_matrix=N             #计算得到法向量矩阵'''
        I=np.zeros(len(self.images))#储存f（12）张图像的每个像素点组成的列向量
        for (x, y), value in np.ndenumerate(self.mask_image):#遍历每个像素点，计算对应法向量
            if value>100.0:#我们只需要处理掩模中的图像部分，其余不用管
                for pos, image in enumerate(self.images):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    I[pos] = image[x, y]#将每图对应的像素点存进去
                #开始计算法向量
                #[dx, dy, dz]
                L=self.light_inform.copy()
                L[:,0]*= I
                L[:,1]*= I
                L[:,2]*= I
                I*=I
                normal_vector, _, _, _ = np.linalg.lstsq(L, I)#最小二乘法计算法向量
                normal_vector/= np.linalg.norm(normal_vector)#归一化平滑处理
            
                if not np.isnan(np.sum(normal_vector)):
                    self.normal_vectors_matrix[x,y]=normal_vector
        print("法向量计算完成")

    def show_image(self,image_matrix):
        
        return 0

    def depth_computation(self):
        rows, cols = np.where(self.mask_image>0.0)#我们只需要计算掩模非0出的像素，即需要合成图像的地方:返回的是矩阵中非零的横坐标向量和综坐标向量,在后期填充构造矩阵是需要用到
        num_nozero_pixels=len(cols)#得到数量
        h_w_matrix= np.zeros(self.mask_image.shape, dtype = np.int)#构造这样一个矩阵利于寻找对应像素点的横纵坐标值
        for pixel_index in range(num_nozero_pixels):
            h_w_matrix[rows[pixel_index],cols[pixel_index]]=pixel_index
        
        #开始构造矩阵并构造下列形式的矩阵等式Mz=v
        M=lil_matrix((2*num_nozero_pixels,num_nozero_pixels))#产生一个稀疏矩阵
        v=np.zeros((2*num_nozero_pixels))

        #填充矩阵
        for pixel_index in range(num_nozero_pixels):#遍历每个像素点求深度信息
            i=rows[pixel_index]
            j=cols[pixel_index]#得到该像素点的行列号信息
        
            n_x=self.normal_vectors_matrix[i,j,0]
            n_y=self.normal_vectors_matrix[i,j,1]
            n_z=self.normal_vectors_matrix[i,j,2]#获取对应的法向量分量
        
            row_idx=(pixel_index-1)*2+1
            if self.mask_image[i+1, j]:
                vertical_index=h_w_matrix[i+1,j]
                v[row_idx]=n_y
                M[row_idx,pixel_index]=-n_z
                M[row_idx,vertical_index]=n_z
            elif self.mask_image[i-1, j]:
                vertical_index=h_w_matrix[i-1, j]
                v[row_idx]=-n_y
                M[row_idx,pixel_index]=-n_z
                M[row_idx,vertical_index]=n_z
             
            # horizontal neighbors
            row_idx=(pixel_index-1)*2+2
            if self.mask_image[i, j+1]:
                horizontal_index=h_w_matrix[i,j+1]
                v[row_idx]=-n_x
                M[row_idx,pixel_index]=-n_z
                M[row_idx,horizontal_index]=n_z
            elif self.mask_image[i, j-1]:
                horizontal_index=h_w_matrix[i,j-1]
                v[row_idx]=n_x
                M[row_idx,pixel_index]=-n_z
                M[row_idx,horizontal_index]=n_z
        
        b=M.T*v
        A=M.T*M
        #构造为Ax=b的形式解方程
        z_depth=scipy.sparse.linalg.lsqr(A, b)[0]#得到深度向量

        #将深度向量转化成矩阵信息
        Z=np.zeros(self.mask_image.shape,dtype=np.float32)
        for pixels_pos in range(num_nozero_pixels):
            i=rows[pixels_pos]
            j=cols[pixels_pos]
            Z[i,j]=z_depth[pixels_pos]

        self.depth_matrix=Z
        print("深度信息计算完成")


    