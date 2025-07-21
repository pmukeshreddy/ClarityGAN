class AdaptiveHyperparams:
    def __init__(self):
        # Initial values
        self.lambda_l1 = 5.0
        self.lambda_gan = 1.0
        self.disc_train_freq = 3
        self.noise_std = 0.05
        
        # Tracking variables
        self.loss_history = {'gen': [], 'disc': []}
        self.disc_accuracy_history = []
        self.stagnation_counter = 0
        
    def update_weights(self, gen_loss, disc_loss, disc_accuracy):
        """Adapt lambda weights based on loss balance"""
        self.loss_history['gen'].append(gen_loss)
        self.loss_history['disc'].append(disc_loss)
        self.disc_accuracy_history.append(disc_accuracy)
        
        # Keep only last 10 epochs for moving average
        if len(self.loss_history['gen']) > 10:
            self.loss_history['gen'] = self.loss_history['gen'][-10:]
            self.loss_history['disc'] = self.loss_history['disc'][-10:]
            self.disc_accuracy_history = self.disc_accuracy_history[-10:]
        
        if len(self.loss_history['gen']) >= 3:
            # Calculate loss ratio trend
            gen_trend = np.mean(self.loss_history['gen'][-3:])
            disc_trend = np.mean(self.loss_history['disc'][-3:])
            loss_ratio = gen_trend / (disc_trend + 1e-8)
            
            # Adjust based on who's winning
            if loss_ratio > 2.0:  # Generator struggling
                self.lambda_gan = min(self.lambda_gan * 1.1, 5.0)
                self.lambda_l1 = max(self.lambda_l1 * 0.95, 1.0)
            elif loss_ratio < 0.5:  # Discriminator struggling  
                self.lambda_gan = max(self.lambda_gan * 0.9, 0.1)
                self.lambda_l1 = min(self.lambda_l1 * 1.05, 15.0)
    
    def update_training_freq(self, disc_accuracy):
        """Adapt discriminator training frequency based on accuracy"""
        if disc_accuracy > 0.85:  # Disc too strong
            self.disc_train_freq = min(self.disc_train_freq + 1, 5)
        elif disc_accuracy < 0.55:  # Disc too weak
            self.disc_train_freq = max(self.disc_train_freq - 1, 1)
    
    def update_noise(self, disc_accuracy, epoch):
        """Adaptive noise based on discriminator performance and training stage"""
        base_noise = 0.05 * (0.95 ** (epoch // 10))  # Decay over time
        
        if disc_accuracy > 0.9:  # Disc overfitting
            self.noise_std = min(base_noise * 2.0, 0.1)
        elif disc_accuracy < 0.6:  # Disc underfitting
            self.noise_std = max(base_noise * 0.5, 0.01)
        else:
            self.noise_std = base_noise
    
    def check_stagnation(self):
        """Detect and respond to training stagnation"""
        if len(self.loss_history['gen']) >= 5:
            recent_gen = self.loss_history['gen'][-5:]
            gen_variance = np.var(recent_gen)
            
            if gen_variance < 0.001:  # Very low variance = stagnation
                self.stagnation_counter += 1
                if self.stagnation_counter >= 3:
                    # Shake things up
                    self.lambda_gan *= 1.5
                    self.noise_std *= 1.2
                    self.stagnation_counter = 0
                    return True
        return False
    
    def get_current_params(self):
        return {
            'lambda_l1': self.lambda_l1,
            'lambda_gan': self.lambda_gan, 
            'disc_train_freq': self.disc_train_freq,
            'noise_std': self.noise_std
        }

def calculate_discriminator_accuracy(disc_outputs_real, disc_outputs_fake):
    """Calculate discriminator accuracy for adaptive updates"""
    with torch.no_grad():
        # Real predictions should be > 0 (after sigmoid > 0.5)
        real_correct = sum([(torch.sigmoid(pred) > 0.5).float().mean() for pred in disc_outputs_real])
        # Fake predictions should be < 0 (after sigmoid < 0.5)  
        fake_correct = sum([(torch.sigmoid(pred) < 0.5).float().mean() for pred in disc_outputs_fake])
        
        total_correct = (real_correct + fake_correct) / (len(disc_outputs_real) + len(disc_outputs_fake))
        return total_correct.item()

# Integration into training loop:
def integrate_adaptive_system():
    """Shows how to integrate into your existing training loop"""
    
    adaptive_hp = AdaptiveHyperparams()
    
    # In your training loop, replace fixed hyperparameters:
    for epoch in range(num_epochs):
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        
        for step, (blur, sharp) in enumerate(train_loader):
            # Get current adaptive parameters
            params = adaptive_hp.get_current_params()
            
            # Use adaptive parameters instead of fixed ones
            lambda_l1 = params['lambda_l1']
            lambda_gan = params['lambda_gan']
            disc_train_freq = int(params['disc_train_freq'])
            noise_std = params['noise_std']
            
            # Your existing training code here...
            # fake = gen(blur)
            
            # Train discriminator with adaptive frequency
            if step % disc_train_freq == 0:
                # Your discriminator training...
                # d_real_preds = disc(blur_noisy, sharp_noisy)
                # d_fake_preds = disc(blur_noisy, fake_noisy)
                pass
            
            # Calculate discriminator accuracy for adaptation
            # disc_accuracy = calculate_discriminator_accuracy(d_real_preds, d_fake_preds)
            
            # Your generator training with adaptive weights...
            # loss_g = (lambda_gan * loss_g_gan) + (lambda_l1 * loss_g_l1) + (lambda_perceptual * loss_g_perceptual)
            
            epoch_gen_loss += 0  # Add your actual gen loss
            epoch_disc_loss += 0  # Add your actual disc loss
        
        # Update adaptive parameters at end of epoch
        avg_gen_loss = epoch_gen_loss / len(train_loader)
        avg_disc_loss = epoch_disc_loss / len(train_loader)
        # avg_disc_accuracy = epoch_disc_accuracy / len(train_loader)
        
        # adaptive_hp.update_weights(avg_gen_loss, avg_disc_loss, avg_disc_accuracy)
        # adaptive_hp.update_training_freq(avg_disc_accuracy)
        # adaptive_hp.update_noise(avg_disc_accuracy, epoch)
        
        # Check for stagnation and respond
        if adaptive_hp.check_stagnation():
            print(f"Epoch {epoch}: Stagnation detected, adjusting hyperparameters")
        
        # Print current adaptive parameters
        current_params = adaptive_hp.get_current_params()
        print(f"Epoch {epoch}: λ_L1={current_params['lambda_l1']:.3f}, "
              f"λ_GAN={current_params['lambda_gan']:.3f}, "
              f"freq={current_params['disc_train_freq']}, "
              f"noise={current_params['noise_std']:.4f}")
