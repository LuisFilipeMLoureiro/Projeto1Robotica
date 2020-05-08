#! /usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


import visao_module


bridge = CvBridge()

cv_image = None

debug_img = None

media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


maior_area = 0.0 # Variavel com a area do maior contorno

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0
xinter=0
yinter=0
frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

#tfl = 0

tf_buffer = tf2_ros.Buffer()


def recebe(msg):
	global x # O global impede a recriacao de uma variavel local, para podermos usar o x global ja'  declarado
	global y
	global z
	global id
	for marker in msg.markers:
		id = marker.id
		marcador = "ar_marker_" + str(id)

		print(tf_buffer.can_transform(frame, marcador, rospy.Time(0)))
		header = Header(frame_id=marcador)
		# Procura a transformacao em sistema de coordenadas entre a base do robo e o marcador numero 100
		# Note que para seu projeto 1 voce nao vai precisar de nada que tem abaixo, a 
		# Nao ser que queira levar angulos em conta
		trans = tf_buffer.lookup_transform(frame, marcador, rospy.Time(0))
		
		# Separa as translacoes das rotacoes
		x = trans.transform.translation.x
		y = trans.transform.translation.y
		z = trans.transform.translation.z
		# ATENCAO: tudo o que vem a seguir e'  so para calcular um angulo
		# Para medirmos o angulo entre marcador e robo vamos projetar o eixo Z do marcador (perpendicular) 
		# no eixo X do robo (que e'  a direcao para a frente)
		t = transformations.translation_matrix([x, y, z])
		# Encontra as rotacoes e cria uma matriz de rotacao a partir dos quaternions
		r = transformations.quaternion_matrix([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
		m = numpy.dot(r,t) # Criamos a matriz composta por translacoes e rotacoes
		z_marker = [0,0,1,0] # Sao 4 coordenadas porque e'  um vetor em coordenadas homogeneas
		v2 = numpy.dot(m, z_marker)
		v2_n = v2[0:-1] # Descartamos a ultima posicao
		n2 = v2_n/linalg.norm(v2_n) # Normalizamos o vetor
		x_robo = [1,0,0]
		cosa = numpy.dot(n2, x_robo) # Projecao do vetor normal ao marcador no x do robo
		angulo_marcador_robo = math.degrees(math.acos(cosa))

		# Terminamos
		print("id: {} x {} y {} z {} angulo {} ".format(id, x,y,z, angulo_marcador_robo))


def linha(frame):
    global xinter
    global yinter
    global debug_img
    
    # frame=cv_image #CV IMAGE TELA PRETA
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # kernel= np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    kernel= np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharp=cv2.filter2D(gray,-1,kernel) 
    #kernel= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    #sharp=cv2.filter2D(sharp,-1,kernel)
    sharp[:-200,:]=0 #deixar uma parte de visao preta

    edges = cv2.Canny(sharp,50,150,apertureSize = 3) 

    debug_img = sharp.copy()

    lines = cv2.HoughLines(edges,1,np.pi/180, 45) #mudar a sensibilidade para 100 

    linha_achada1=False  
    linha_achada=False       
    h1=0
    m1=0
    m2=0
    h2=0

    if lines is None:
        return frame, (-1,-1)

    for linha in lines: #usamos como referencia: https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
        for r,theta in linha: 
            a = np.cos(theta) 
            b = np.sin(theta)  
            x0 = a*r 
            y0 = b*r 
            x1 = int(x0 + 1000*(-b))  
            y1 = int(y0 + 1000*(a))  
            x2 = int(x0 - 1000*(-b))  
            y2 = int(y0 - 1000*(a)) 
            if (x2-x1)!=0:
                coef1=(y2-y1)/(x2-x1)
            if coef1 < -0.86 and coef1 > -10:
                if linha_achada1==False:
                    linha_achada1=True
                    m1=coef1
                    h1=(y1-coef1*x1)
                    cv2.line(frame,(x1,y1), (x2,y2), (0,0,255),2)
            elif coef1 > 0.86 and coef1<10:
                if linha_achada==False:
                    linha_achada=True
                    m2=coef1
                    h2=(y1-coef1*x1)
                    cv2.line(frame,(x1,y1), (x2,y2), (0,0,255),2)
            
            if (m1-m2)!=0:                    
                xinter=int((h2-h1)/(m1-m2))
            yinter=int(m1*xinter + h1) 
            cv2.circle(frame,(xinter,yinter), 10, (0,255,0), -1)


            print("XINTER {0}".format((xinter,yinter)))
            
            #cv2.imshow('linhas',frame)
    return frame, (xinter,yinter)




# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global resultados
    global maior_area

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        antes = time.clock()
        temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados
        centro, saida_net, resultados =  visao_module.processa(temp_image)  
        #media, centro, maior_area =  cormodule.identifica_cor(cv_image)      
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        depois = time.clock()
        print("CENTRO {0}".format(centro))
        # Desnecessário - Hough e MobileNet já abrem janelas
        cv_image = saida_net.copy()
    except CvBridgeError as e:
        print('ex', e)
    
if __name__=="__main__":
    rospy.init_node("cor")

    topico_imagem = "/camera/rgb/image_raw/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    recebedor = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, recebe) # Para recebermos notificacoes de que marcadores foram vistos


    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    # Conserto linha vermelha
    tfl = tf2_ros.TransformListener(tf_buffer)

    tolerancia = 25

    # Exemplo de categoria de resultados
    # [('chair', 86.965459585189819, (90, 141), (177, 265))]

    try:
        # Inicializando - por default gira no sentido anti-horário
        # vel = Twist(Vector3(0,0,0), Vector3(0,0,math.pi/10.0))
        
        while not rospy.is_shutdown():
            for r in resultados:
                print(r)
            #velocidade_saida.publish(vel)


            if cv_image is not None:
                # Note que o imshow precisa ficar *ou* no codigo de tratamento de eventos *ou* no thread principal, não em ambos
                #cv2.imshow("cv_image no loop principal", cv_image)
                imagem, pontointer= linha(cv_image)
                cv2.imshow("Linha", imagem)
                if debug_img is not None:
                    cv2.imshow("DEBUG", debug_img)

                if xinter>centro[0]:
                    vel = Twist(Vector3(0.7,0,0), Vector3(0,0,-0.05))
                    velocidade_saida.publish(vel)
        
                elif xinter<centro[0]:
                    vel = Twist(Vector3(0.7,0,0), Vector3(0,0,0.05))
                    velocidade_saida.publish(vel)
                elif xinter<=0 or yinter<= 0:
                    vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
                    velocidade_saida.publish(vel)

                
                
                
                cv2.waitKey(1)
                
            rospy.sleep(0.1)
                    

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")