# Keras imports
from keras.models import *
from keras.layers import *
from keras import backend as K

IMAGE_ORDERING = 'channels_last' 

# Paper: https://arxiv.org/pdf/1505.04597.pdf)%E5%92%8C%5bTiramisu%5d(https://arxiv.org/abs/1611.09326.pdf

def build_unet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None):

	# CONTRACTING PATH

	img_input = Input(shape=img_shape)

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x

	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv7', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv8', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv9', data_format=IMAGE_ORDERING)(x)
	x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv10', data_format=IMAGE_ORDERING)(x)
	f5 = x
	
	#### DECONTRACTING PATH

	o = Conv2DTranspose(512, kernel_size=(4,4), strides=(2,2), name='conv11', data_format=IMAGE_ORDERING)(o)
	o = concatenate([f4, o], axis=1 )
	o = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv12', data_format=IMAGE_ORDERING)(o)
	o = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv13', data_format=IMAGE_ORDERING)(o)

	o = Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), name='conv14', data_format=IMAGE_ORDERING)(o)
        o = concatenate([f3, o], axis=1 )
	o = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv15', data_format=IMAGE_ORDERING)(o)
	o = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv16', data_format=IMAGE_ORDERING)(o)

	o = Conv2DTranspose(128, kernel_size=(4,4), strides=(2,2), name='conv17', data_format=IMAGE_ORDERING)(o)
        o = concatenate([f2, o], axis=1 )
	o = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv18', data_format=IMAGE_ORDERING)(o)
	o = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv19', data_format=IMAGE_ORDERING)(o)

	o = Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), name='conv20', data_format=IMAGE_ORDERING)(o)
        o = concatenate([f1, o], axis=1 )
	o = Conv2D(64, (3, 3), activation='relu', padding='same',  name='conv21', data_format=IMAGE_ORDERING)(o)
	o = Conv2D(64, (3, 3), activation='relu', padding='same',  name='conv22', data_format=IMAGE_ORDERING)(o)

	o = Conv2D(filters=nclasses, kernel_size=(1,1), padding='same', data_format=IMAGE_ORDERING)(o)
	
	# Ensure that output has the same size HxW as input and apply softmax to compute pixel prob.
        o_shape = Model(img_input, o).output_shape
        o = Cropping2D(((o_shape[1] - img_shape[0]) / 2, (o_shape[2] - img_shape[1]) / 2), name='crop')(o)

        """
        o_shape = Model(img_input, o).output_shape
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]
        o = (Reshape((-1, outputHeight * outputWidth)))(o)
        o = (Permute((2, 1)))(o)
        """

        # Reshape to vector
        curlayer_output_shape = Model(inputs=img_input, outputs=o).output_shape
        if K.image_dim_ordering() == 'tf':
            outputHeight = curlayer_output_shape[1]
            outputWidth = curlayer_output_shape[2]
        else:
            outputHeight = curlayer_output_shape[2]
            outputWidth = curlayer_output_shape[3]
        o = Reshape(target_shape=(outputHeight * outputWidth, nclasses))(o)

        o = Activation('softmax')(o)
        model = Model(img_input, o)
        #model.outputWidth = outputWidth
        #model.outputHeight = outputHeight

        # Freeze some layers
        if freeze_layers_from is not None:
            freeze_layers(model, freeze_layers_from)

        return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [224, 224, 3]
    print (' > Building')
    model = build_fcn8(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
