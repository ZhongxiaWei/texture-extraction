// Advanced texture classification/feature extraction
// Name:Zhongxia Wei
// ID:9443118655
// email:zhongxiw@usc.edu
// Compiled on mac with xcode

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


using namespace std;


void TransToFloat (unsigned char (*imageData)[128][128], float (*imageDataF)[128][128]);
void Wall (float (*imageData)[128][128], float (*imageDataWall)[132][132]);
void GetSubLoc (float (*imageDataW)[132][132], float(*subLoc)[128][128]);
void GetCha (float (*subLoc)[132][132], double (*chaVec)[25]);
float* crossProduct(float a[5], float b[5]);

int main(int argc, const char * argv[]) {
    string grassFile[48] = {
        "grass_01.raw","grass_02.raw","grass_03.raw","grass_04.raw","grass_05.raw",
        "grass_06.raw","grass_07.raw","grass_08.raw","grass_09.raw","grass_10.raw",
        "grass_11.raw","grass_12.raw","grass_13.raw","grass_14.raw","grass_15.raw",
        "grass_16.raw","grass_17.raw","grass_18.raw","grass_19.raw","grass_20.raw",
        "grass_21.raw","grass_22.raw","grass_23.raw","grass_24.raw","grass_25.raw",
        "grass_26.raw","grass_27.raw","grass_28.raw","grass_29.raw","grass_30.raw",
        "grass_31.raw","grass_32.raw","grass_33.raw","grass_34.raw","grass_35.raw",
        "grass_36.raw","grass_37.raw","grass_38.raw","grass_39.raw","grass_40.raw",
        "grass_41.raw","grass_42.raw","grass_43.raw","grass_44.raw","grass_45.raw",
        "grass_46.raw","grass_47.raw","grass_48.raw"};
    string leatherFile[48] = {
        "leather_01.raw","leather_02.raw","leather_03.raw","leather_04.raw","leather_05.raw",
        "leather_06.raw","leather_07.raw","leather_08.raw","leather_09.raw","leather_10.raw",
        "leather_11.raw","leather_12.raw","leather_13.raw","leather_14.raw","leather_15.raw",
        "leather_16.raw","leather_17.raw","leather_18.raw","leather_19.raw","leather_20.raw",
        "leather_21.raw","leather_22.raw","leather_23.raw","leather_24.raw","leather_25.raw",
        "leather_26.raw","leather_27.raw","leather_28.raw","leather_29.raw","leather_30.raw",
        "leather_31.raw","leather_32.raw","leather_33.raw","leather_34.raw","leather_35.raw",
        "leather_36.raw","leather_37.raw","leather_38.raw","leather_39.raw","leather_40.raw",
        "leather_41.raw","leather_42.raw","leather_43.raw","leather_44.raw","leather_45.raw",
        "leather_46.raw","leather_47.raw","leather_48.raw"};
    string  sandFile[48] = {
        "sand_01.raw","sand_02.raw","sand_03.raw","sand_04.raw","sand_05.raw",
        "sand_06.raw","sand_07.raw","sand_08.raw","sand_09.raw","sand_10.raw",
        "sand_11.raw","sand_12.raw","sand_13.raw","sand_14.raw","sand_15.raw",
        "sand_16.raw","sand_17.raw","sand_18.raw","sand_19.raw","sand_20.raw",
        "sand_21.raw","sand_22.raw","sand_23.raw","sand_24.raw","sand_25.raw",
        "sand_26.raw","sand_27.raw","sand_28.raw","sand_29.raw","sand_30.raw",
        "sand_31.raw","sand_32.raw","sand_33.raw","sand_34.raw","sand_35.raw",
        "sand_36.raw","sand_37.raw","sand_38.raw","sand_39.raw","sand_40.raw",
        "sand_41.raw","sand_42.raw","sand_43.raw","sand_44.raw","sand_45.raw",
        "sand_46.raw","sand_47.raw","sand_48.raw"};
    string strawFile[48] = {
        "straw_01.raw","straw_02.raw","straw_03.raw","straw_04.raw","straw_05.raw",
        "straw_06.raw","straw_07.raw","straw_08.raw","straw_09.raw","straw_10.raw",
        "straw_11.raw","straw_12.raw","straw_13.raw","straw_14.raw","straw_15.raw",
        "straw_16.raw","straw_17.raw","straw_18.raw","straw_19.raw","straw_20.raw",
        "straw_21.raw","straw_22.raw","straw_23.raw","straw_24.raw","straw_25.raw",
        "straw_26.raw","straw_27.raw","straw_28.raw","straw_29.raw","straw_30.raw",
        "straw_31.raw","straw_32.raw","straw_33.raw","straw_34.raw","straw_35.raw",
        "straw_36.raw","straw_37.raw","straw_38.raw","straw_39.raw","straw_40.raw",
        "straw_41.raw","straw_42.raw","straw_43.raw","straw_44.raw","straw_45.raw",
        "straw_46.raw","straw_47.raw","straw_48.raw"};
    

    double grassChaVec[48][25];
    double leatherChaVec[48][25];
    double sandChaVec[48][25];
    double strawChaVec[48][25];
    double normGrass[48][25];
    double normLeather[48][25];
    double normSand[48][25];
    double normStraw[48][25];
    
//grass characeristic vector
    for (int p=0; p<48; p++) {
        FILE *file;
        unsigned char data[128][128];
        float DataF[128][128];
        float DataW[132][132];
        float subLoc[128][128];
        float subLocW[132][132];
        double chaVec[25];
        
        file = fopen(grassFile[p].c_str(), "rb");
        fread(data, sizeof(unsigned char), 128*128, file);
        TransToFloat(&data, &DataF);
        Wall(&DataF, &DataW);
        GetSubLoc(&DataW, &subLoc);
        Wall(&subLoc, &subLocW);
        GetCha(&subLocW, &chaVec);
        for (int v=0; v<25; v++) {
            grassChaVec[p][v]=chaVec[v];
        }
    }
    // leather chaceristic vector
    for (int p=0; p<48; p++) {
        FILE *file;
        unsigned char data[128][128];
        float DataF[128][128];
        float DataW[132][132];
        float subLoc[128][128];
        float subLocW[132][132];
        double chaVec[25];
        
        file = fopen(leatherFile[p].c_str(), "rb");
        fread(data, sizeof(unsigned char), 128*128, file);
        TransToFloat(&data, &DataF);
        Wall(&DataF, &DataW);
        GetSubLoc(&DataW, &subLoc);
        Wall(&subLoc, &subLocW);
        GetCha(&subLocW, &chaVec);
        for (int v=0; v<25; v++) {
            leatherChaVec[p][v]=chaVec[v];
        }
    }
    //grass characeristic vector
    for (int p=0; p<48; p++) {
        FILE *file;
        unsigned char data[128][128];
        float DataF[128][128];
        float DataW[132][132];
        float subLoc[128][128];
        float subLocW[132][132];
        double chaVec[25];
        
        file = fopen(sandFile[p].c_str(), "rb");
        fread(data, sizeof(unsigned char), 128*128, file);
        TransToFloat(&data, &DataF);
        Wall(&DataF, &DataW);
        GetSubLoc(&DataW, &subLoc);
        Wall(&subLoc, &subLocW);
        GetCha(&subLocW, &chaVec);
        for (int v=0; v<25; v++) {
            sandChaVec[p][v]=chaVec[v];
        }
    }
    //grass characeristic vector
    for (int p=0; p<48; p++) {
        FILE *file;
        unsigned char data[128][128];
        float DataF[128][128];
        float DataW[132][132];
        float subLoc[128][128];
        float subLocW[132][132];
        double chaVec[25];
        
        file = fopen(strawFile[p].c_str(), "rb");
        fread(data, sizeof(unsigned char), 128*128, file);
        TransToFloat(&data, &DataF);
        Wall(&DataF, &DataW);
        GetSubLoc(&DataW, &subLoc);
        Wall(&subLoc, &subLocW);
        GetCha(&subLocW, &chaVec);
        for (int v=0; v<25; v++) {
            strawChaVec[p][v]=chaVec[v];
        }
    }
    
    
    //normalize
    double max[25];
    double min[25];
    for (int v=0; v<25; v++) {
        max[v]=0;
        min[v]=100000000;
        for (int p=0; p<192; p++) {
            if (p/48==0) {
                if (max[v]<grassChaVec[p][v]) {
                    max[v]=grassChaVec[p][v];
                }
                if (min[v]>grassChaVec[p][v]) {
                    min[v]=grassChaVec[p][v];
                }
            }
            else if (p/48==1){
                if (max[v]<leatherChaVec[p%48][v]) {
                    max[v]=leatherChaVec[p%48][v];
                }
                if (min[v]>leatherChaVec[p%48][v]) {
                    min[v]=leatherChaVec[p%48][v];
                }
            }
            else if (p/48==2){
                if (max[v]<sandChaVec[p%48][v]) {
                    max[v]=sandChaVec[p%48][v];
                }
                if (min[v]>sandChaVec[p%48][v]) {
                    min[v]=sandChaVec[p%48][v];
                }
            }
            else{
                
                if (max[v]<strawChaVec[p%48][v]) {
                    max[v]=strawChaVec[p%48][v];
                }
                if (min[v]>strawChaVec[p%48][v]) {
                    min[v]=strawChaVec[p%48][v];
                }
            }
        }
    }
    
    for (int p=0; p<48; p++) {
        for (int v=0; v<25; v++) {
            normGrass[p][v]=(grassChaVec[p][v]-min[v])/(max[v]-min[v]);
            normLeather[p][v]=(leatherChaVec[p][v]-min[v])/(max[v]-min[v]);
            normSand[p][v]=(sandChaVec[p][v]-min[v])/(max[v]-min[v]);
            normStraw[p][v]=(strawChaVec[p][v]-min[v])/(max[v]-min[v]);
        }
    }
    
    
    ofstream data;
    data.open("character.csv", ios::out|ios::trunc);
    for (int p=0 ; p<48; p++) {
        for (int v=0; v<25; v++) {
            data<<(normGrass[p][v])<<","<<endl;
        }
    }
    
    for (int p=0 ; p<48; p++) {
        for (int v=0; v<25; v++) {
            data<<(normLeather[p][v])<<","<<endl;
        }
    }
    
    for (int p=0 ; p<48; p++) {
        for (int v=0; v<25; v++) {
            data<<(normSand[p][v])<<","<<endl;
        }
    }
    
    for (int p=0 ; p<48; p++) {
        for (int v=0; v<25; v++) {
            data<<(normStraw[p][v])<<","<<endl;
        }
    }
    
    
    data.close();

    
}


void GetCha (float (*subLoc)[132][132], double (*chaVec)[25]){
    float filter[25] = {1,4,6,4,1,-1,-2,0,2,1,-1,0,2,0,-1,-1,2,0,-2,1,1,-4,6,-4,1};
    int chaVecC=0;
    for (int fc1 = 0; fc1<5; fc1++) {
        for (int fc2 = 0; fc2<5; fc2++) {
            double intensity=0;
            float filter1[5]=
            {filter[fc1*5],filter[fc1*5+1],filter[fc1*5+2],filter[fc1*5+3],filter[fc1*5+4]};
            float filter2[5]=
            {filter[fc2*5],filter[fc2*5+1],filter[fc2*5+2],filter[fc2*5+3],filter[fc2*5+4]};
            float *filter2D=crossProduct(filter1, filter2);
            //                cout<<filter2D[0]<<filter2D[2]<<endl;
            //apply filter to subLoc
            for (int i=0; i<128 ; i++) {
                for (int j=0; j<128 ; j++) {
                    int m=i+2;
                    int n=j+2;
                    int fc=0;
                    float sum=0;
                    for (int x=m-2; x<m+3; x++) {
                        for (int y=n-2; y<n+3; y++) {
                            sum = (*subLoc)[x][y] * filter2D[fc] + sum;
                            fc++;
                        }
                    }
                    intensity=sum*sum+intensity;
                }
            }
            (*chaVec)[chaVecC]=intensity/(128*128);
            chaVecC++;
        }
    }
}

void GetSubLoc (float (*imageDataW)[132][132], float(*subLoc)[128][128]){
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            int m=i+2;
            int n=j+2;
            float sum=0;
            float locMean;
            for (int x=m-2; x<m+2; x++) {
                for (int y=n-2; y<n+2; y++) {
                    sum=sum+(*imageDataW)[x][y];
                }
            }
            locMean=sum/25;
            (*subLoc)[i][j]=(*imageDataW)[m][n]-locMean;
        }
    }
}


void TransToFloat (unsigned char (*imageData)[128][128], float (*imageDataF)[128][128]){
    for (int row=0; row<128; row++) {
        for (int col=0; col<128; col++) {
            (*imageDataF)[row][col]=(*imageData)[row][col];
        }
    }
}



void Wall (float (*imageData)[128][128], float (*imageDataWall)[132][132]){
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            (*imageDataWall)[i+2][j+2] = (*imageData)[i][j];
        }
    }
    //copy first column
    for (int i=0 ; i<128; i++) {
        (*imageDataWall)[i+2][1] = (*imageData)[i][1];
    }
    //copy sec column
    for (int i=0 ; i<128; i++) {
        (*imageDataWall)[i+2][0] = (*imageData)[i][2];
    }
    //copy last column
    for (int i=0 ; i<128; i++) {
        (*imageDataWall)[i+2][128+2] = (*imageData)[i][128-2];
    }
    //copy penult column
    for (int i=0 ; i<128; i++) {
        (*imageDataWall)[i+2][128+3] = (*imageData)[i][128-3];
    }
    //copy first row
    for (int j=0 ; j<128+4; j++) {
        (*imageDataWall)[1][j] = (*imageDataWall)[3][j];
    }
    //copy sec row
    for (int j=0 ; j<128+4; j++) {
        (*imageDataWall)[0][j] = (*imageDataWall)[4][j];
    }
    //copy last row
    for (int j=0 ; j<128+4; j++) {
        (*imageDataWall)[128+2][j] = (*imageDataWall)[128][j];
    }
    //copy penult row
    for (int j=0 ; j<128+4; j++) {
        (*imageDataWall)[128+3][j] = (*imageDataWall)[128-1][j];
    }
}

float* crossProduct(float a[5], float b[5]){
    float* res = new float[25];
    for (int i=0; i<5 ;i++) {
        for (int j=0; j<5; j++) {
            res[i*5+j]=a[i]*b[j];
        }
    }
    return res;
}
