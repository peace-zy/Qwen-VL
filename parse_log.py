import sys
import matplotlib.pyplot as plt

def main():
    infile = sys.argv[1]

    loss = []
    lr = []
    epoch = []
    MERGE = True
    USE_STEP = True
    #{'loss': 0.4765, 'learning_rate': 8.653946898401348e-06, 'epoch': 0.02}
    with open(infile, 'r') as f:
        for line in f:
            if "{'loss':" in line:
                data = eval(line.strip())
                print(data)
                if USE_STEP:
                    loss.append(data["loss"])
                    lr.append(data["learning_rate"])
                else:
                    if len(epoch) > 0 and data["epoch"] == epoch[-1]:
                        continue
                    loss.append(data["loss"])
                    lr.append(data["learning_rate"])
                    epoch.append(data["epoch"])
    #print(lr[0:400])
    gradient_accumulation_steps = 16
    gradient_accumulation_steps = 1
    if MERGE == True:
        #plot 1
        plt.subplot(2, 1, 1)
        if USE_STEP:
            plt.plot([i for i in range(1,len(lr) + 1, gradient_accumulation_steps)], lr[::gradient_accumulation_steps], label="train lr")
            plt.xlabel('step')
        else:
            plt.plot(epoch, lr, label="train lr")
            plt.xlabel('epoch')
        plt.title("Model lr")
        plt.ylabel('lr')

        #plot 2
        plt.subplot(2, 1, 2)
        if USE_STEP:
            plt.plot([i for i in range(1,len(lr) + 1, gradient_accumulation_steps)], loss[::gradient_accumulation_steps], label="train loss")
            plt.xlabel('step')
        else:
            plt.plot(epoch, loss, label="train loss")
            plt.xlabel('epoch')
       
        plt.title("Model loss")
        plt.ylabel('loss')
        
        plt.suptitle("Train lr & loss")
        plt.tight_layout()
        #plt.savefig("train_lr_loss_llava_zero3_cosine_a800.png")
        plt.savefig("train_lr_loss_sekuai_zero2_cosine_a800.png")
        #plt.savefig("train_lr_loss_new.png")
        #plt.savefig("train_lr_loss.png")

    else:
        #plot 1
        plt.figure(1)
        if USE_STEP:
            plt.plot([i for i in range(1,len(lr) + 1)], lr, label="train lr")
            plt.xlabel('step')
        else:
            plt.plot(epoch, lr, label="train lr")
            plt.xlabel('epoch')
        plt.title("Model lr")
        plt.ylabel('lr')
        plt.savefig("lr.png")

        #plot 2
        plt.figure(2)
        if USE_STEP:
            plt.plot([i for i in range(1,len(lr) + 1)], loss, label="train loss")
            plt.xlabel('step')
        else:
            plt.plot(epoch, loss, label="train loss")
            plt.xlabel('epoch')
       
        plt.title("Model loss")
        plt.ylabel('loss')
    
        plt.savefig("loss.png")

            
    return

if __name__ == '__main__':
    main()
