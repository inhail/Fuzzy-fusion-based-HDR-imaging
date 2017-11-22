%from RGB to HSV, and check H,S,V chennel separately
inputImg=imread('image01.jpg');
hv=rgb2hsv(inputImg); 
V = hv(:,:,3);

total_pix = size(V,1)*size(V,2);%total number of pixels
histV = imhist(V);%histogram of V channel
% histV��256 x 1���x�}�A�B�C��element�����O�b0-1����
histVsum = cumsum(histV);%cumulative sum of histogram
histVsum_n = histVsum / total_pix;
%partial formula:sigma(histogram))/total_pixel

%�Q���Dw�ȭn����Xr(gamma)
%paper��gamma�ȬOdark region��gray level�[�`�A���W������gray level�[�`
V_norm = V*255;
Vnorm_sum = sum(V_norm(:)); %sum of normalized intensity
dark_sum = 0;
for i = 1:size(V,1)
    for j = 1:size(V,2)
        if V_norm(i,j) > 0 & V_norm(i,j) <= 64 %the proposed dark region 
            %however, we can change a&b value to see the difference
            dark_sum = dark_sum + V_norm(i,j);%�u�N0-64��gray level�[�_��
        end
    end
end
gam_ma = dark_sum / Vnorm_sum;
% We use RANSAC method to find alpha and beta value
% according to the Fig.2 the points are summarized as parameter pts
% RANSAC settings: iteration number is 60, acceptable inlier distance is
% 0.1, inlier threshold is 0.9
pts = [0,0.24,0.33,0.38,0.39,0.4,0.54,0.585,0.73,0.88;0.1,0.185,0.29,0.18,0.29,0.29,0.28,0.375,0.35,0.48];
[al_pha, be_ta] = ransac(pts,60,0.1,0.9);
w = al_pha * gam_ma + be_ta; % w is weight factor

%calculate the threthold
%argmin���N�q�N�O��X�̤p�Ȫ������Ǽ�
abs_value = zeros(256,1);
for i = 1:256
    abs_value(i,1) = abs(w-histVsum_n(i,1));
end
[minNum, minIndex] = min(abs_value);
tor = minIndex / 256;%tor is threshold

%�M�p�q���P���a��O:
%�p�q���Nv channel����under�Mover�ⳡ���A�A�Q�γo2�����h�������
%�t��(preunderimg)�O0~tor�A�j�󵥩�tor���]�w��tor
%�G��(preoverimg)�Otor~1�A�p�󵥩�tor���]�w��0
preoverimg = V;
preunderimg = V;
for i = 1:size(V,1)
    for j = 1:size(V,2)
        if V(i,j) <= tor
            preoverimg(i,j) = 0;
        elseif V(i,j) > tor
            preunderimg(i,j) = tor;
        end
    end
end

%check two subset histograms
%figure('name','subset histograms')
%subplot(2,1,1),imhist(preunderimg)
%subplot(2,1,2),imhist(preoverimg)

%find non-zero min of preunderimg
min_underimg = min(min(preunderimg));

% find max and min of preoverimg
min_overimg = min(min(preoverimg));
max_overimg = max(max(preoverimg));

%linear stretching of preunderimg
for i = 1:size(V,1)
    for j = 1:size(V,2)
        preunderimg(i,j) = (preunderimg(i,j)-min_underimg) / (tor-min_underimg);
    end
end
%linear stretching of preoverimg
for i = 1:size(V,1)
    for j = 1:size(V,2)
        preoverimg(i,j) =(preoverimg(i,j)-min_overimg) / (max_overimg-min_overimg);
    end
end

%�ˬdstretch�᪺histogram
%figure('name','after linear stretching')
%subplot(2,1,1),imhist(preunderimg)
%subplot(2,1,2),imhist(preoverimg)

%Perform CLAHE
overimg = adapthisteq(preoverimg);
underimg = adapthisteq(preunderimg);

% combine H,S and V channel to see under-exposed image
H0(:,:,1) = hv(:,:,1);
H0(:,:,2) = hv(:,:,2);
H0(:,:,3) = underimg;
RGBunderimg = hsv2rgb(H0);
figure('name','underexposed'),imshow(RGBunderimg);
imwrite(RGBunderimg,'under_afterCLAHE.jpg');

% combine H,S and V channel to see over-exposed image
H1(:,:,1) = hv(:,:,1);
H1(:,:,2) = hv(:,:,2);
H1(:,:,3) = overimg;
RGBoverimg = hsv2rgb(H1);
figure('name','overexposed'),imshow(RGBoverimg);
imwrite(RGBoverimg,'over_afterCLAHE.jpg');

%fuzzy fusion
% input 1 :local min-max difference of V channels
%local min-max difference:�HV(i,j)�����ߡA�[�W�P��8��pixel�@9�ӡA�����@��local
%�A�γo��local area�̳̤j�ȴ�̤p��
% input 2 :normalized pixel values of V channels

% input 1 of normal-exposed image
difference_ne = zeros(size(V,1),size(V,2));
for i = 1:size(V,1)
    for j = 1:size(V,2)
        min_ne = V(i,j);
        max_ne = V(i,j);
        for x = 0:2
            for y = 0:2
                if i - x > 0 & j+y <= size(V,2)
                    if V(i-x,j+y) > max_ne
                        max_ne = V(i-x,j+y);
                    elseif V(i-x,j+y) < min_ne
                        min_ne = V(i-x,j+y);
                    end
                end
            end
        end
        difference_ne(i,j) = max_ne - min_ne;
    end
end
input1_ne = difference_ne ;
input2_ne = V ;
fismat_ne = readfis('fuzzy');
output_ne0 = evalfis([input1_ne(:),input2_ne(:)],fismat_ne);
output_ne = reshape(output_ne0,size(V,1),size(V,2));

% input 1 of under-exposed image
underimg_norm = underimg;
difference_ue = zeros(size(V,1),size(V,2));
for i = 1:size(V,1)
    for j = 1:size(V,2)
        min_ne = underimg_norm(i,j);
        max_ne = underimg_norm(i,j);
        for x = 0:2
            for y = 0:2
                if i - x > 0 & j+y <= size(V,2)
                    if underimg_norm(i-x,j+y) > max_ne
                        max_ne = underimg_norm(i-x,j+y);
                    elseif underimg_norm(i-x,j+y) < min_ne
                        min_ne = underimg_norm(i-x,j+y);
                    end
                end
            end
        end
        difference_ue(i,j) = max_ne - min_ne;
    end
end
input1_ue = difference_ue ;
input2_ue = underimg ;
fismat_ue = readfis('fuzzy');
output_ue0 = evalfis([input1_ue(:),input2_ue(:)],fismat_ue);
output_ue = reshape(output_ue0,size(V,1),size(V,2));

% input 1 of over-exposed image
overimg_norm = overimg;
difference_oe = zeros(size(V,1),size(V,2));
for i = 1:size(V,1)
    for j = 1:size(V,2)
        min_oe = overimg_norm(i,j);
        max_oe = overimg_norm(i,j);
        for x = 0:2
            for y = 0:2
                if i - x > 0 & j+y <= size(V,2)
                    if overimg_norm(i-x,j+y) > max_oe
                        max_oe = overimg_norm(i-x,j+y);
                    elseif overimg_norm(i-x,j+y) < min_oe
                        min_oe = overimg_norm(i-x,j+y);
                    end
                end
            end
        end
        difference_oe(i,j) = max_oe - min_oe;
    end
end
input1_oe = difference_oe ;
input2_oe = overimg ;
fismat_oe = readfis('fuzzy');
output_oe0 = evalfis([input1_oe(:),input2_oe(:)],fismat_oe);
output_oe = reshape(output_oe0,size(V,1),size(V,2)); 

%���Mpaper���y�{�ϬO���NHSV�T�ӦX�_�ӦA�୼RGB�A�M����O�h��R,G�MB��fuzzy
%���o�˥X�Ӫ����G�|�D�`�t�S��
%�ҥH�ⰵ��CLAHE���T��V channel������fuzzy fusion��A�A�MHS�X�_��
%�̫�A�নRGB��X
for i = 1:size(V,1)
    for j = 1:size(V,2)
        down = output_ne(i,j) + output_ue(i,j) + output_oe(i,j); 
        Vup = (V(i,j)*output_ne(i,j)) + (underimg(i,j)*output_ue(i,j)) + (overimg(i,j)*output_oe(i,j));
        preHDRimg(i,j) = Vup / down ;
    end
end

HDRimg(:,:,1) = hv(:,:,1);
HDRimg(:,:,2) = hv(:,:,2);
HDRimg(:,:,3) = preHDRimg;
RGBHDRimg = hsv2rgb(HDRimg);
figure('name','HDR'),imshow(RGBHDRimg);
imwrite(RGBHDRimg,'HDR.jpg');