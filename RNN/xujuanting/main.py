1# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:05:50 2018

@author: janti
"""
import tensorflow as tf
import Setup as Set
#import moving_filter as filt
import RNN_prediction as RNN_pre
import velocity_to_power as v2p
import numpy as np
from matplotlib import pyplot as plt
import xlrd

# In[]
sess = tf.Session()  
  
# First, load meta graph and restore weights  
saver = tf.train.import_meta_graph('model/model.ckpt-50000.meta')  
saver.restore(sess, tf.train.latest_checkpoint('./'))  
  
# Second, access and create placeholders variables and create feed_dict to feed new data  
graph = tf.get_default_graph()  
x = graph.get_tensor_by_name('x:0')  
y = graph.get_tensor_by_name('y:0')  


# In[]
##load profiles for different driving cycles: JC08 UDDS EUDC
velocity="JC08_velocity.xlsx"
book= xlrd.open_workbook (velocity, encoding_override = "utf-8")
data_sheet = book.sheet_by_index(0)
velocity= np.asarray([data_sheet.col_values(0)]).T
test_X=velocity

m=len(velocity)
##声明变量
power_bat=np.zeros((m+10,1))
power_oc=np.zeros((m+10,1))
I_c=np.zeros((m,1))
I_b=np.zeros((m,1))
U_uc=np.zeros((m,1))
U_bat=np.zeros((m,1))
soc_bat=np.zeros((m,1))

# In[]
#step 2 setup battery pack ,UC pack,and DC-DC converter,PV modular
R_s,U_ocv,soc_initial =Set.setup_battery()
Cap_UC,U_uc[0],r_l,r_s = Set.setup_UC()
U_ocv=np.array(U_ocv)
## step 3 global initial value
n_bat=3    ##  n组电池，n应该为2的n次幂
n_uc=5  #电容组数
tick = 1  #%采样率，或者更新时间
Q = 12.5 * 3600    #%电池电量  mAh
## step 4 charge the load 
soc_bat[0]=soc_initial      #%初始的SOC
U_bat[0],Rs_bat=Set.cal_bat_RS_U (U_ocv,R_s,soc_bat[0])     #%初始的U_bat和Rs_bat
#U_uc[0]=Set.cal_UC(U_uc[0],tick,I_c,Cap_UC)              #%初始的电容电压值(单体)
power_load=v2p.vel_to_power(velocity)

for i in range (m): 
    if i<10:
        power_bat[i],power_oc[i]=0.5*power_load[i],0.5*power_load[i]
    else:
        if i%10==0:
            feed_dict = {x:test_X[i-10:i], y:test_X[i:i+10]}  
            # Access the op that want to run 
            op_to_restore = graph.get_tensor_by_name('op_to_restore:0') 
            predicted_X=sess.run(op_to_restore, feed_dict)
            print (predicted_X) 
           # predicted_X=[[pred] for pred in RNN_pre.regressor.predict(test_X[i-10:i])]
            power_bat[i:i+10]=v2p.vel_to_power(predicted_X)
            power_oc[i:i+10]=power_load[i:i+10]- power_bat[i:i+10]
            
    I_c[i]=power_oc[i]/U_uc[i]
    U_uc[i+1]=Set.cal_UC(U_uc[i],tick,I_c[i],Cap_UC); #电容的电压值 
    soc_bat[i+1]=Set.cal_battery(soc_bat[i],I_b[i],tick,Q)
    U_bat[i+1],Rs_bat=Set.cal_bat_RS_U(U_ocv,R_s,soc_bat[i+1])
    
   

print("variation of I_b: ",np.var(I_b))

# In[]
        
fig=plt.figure()
plt.plot(power_load,'g',power_oc*n_uc,'r',power_bat*n_bat,'b')
plt.xlabel("Time [s]",fontsize=10)
plt.ylabel("Power [w/s]",fontsize=10)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#fig.set_size_inches(20, 7.5)
plt.show()    

fig=plt.figure()
plt.plot(soc_bat,'b')
plt.xlabel("Time [s]",fontsize=10)
plt.ylabel("soc_bat",fontsize=10)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#fig.set_size_inches(20, 7.5)
plt.show()  

fig=plt.figure()
plt.plot(I_b,'b',I_c,'r')
plt.xlabel("Time [s]",fontsize=10)
plt.ylabel("current",fontsize=10)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#fig.set_size_inches(20, 7.5)
plt.show()    
        
fig=plt.figure()
plt.plot(U_bat,'b')
plt.xlabel("Time [s]",fontsize=10)
plt.ylabel("Voltage",fontsize=10)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#fig.set_size_inches(20, 7.5)
plt.show() 

fig=plt.figure()
plt.plot(U_uc,'r')
plt.xlabel("Time [s]",fontsize=10)
plt.ylabel("Voltage",fontsize=10)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#fig.set_size_inches(20, 7.5)
plt.show()         
