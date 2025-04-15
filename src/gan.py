import torch
import torch.nn as nn
import random

# Define the Generator
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Hidden dimension of LSTM layers.
            max_seq_len (int): Maximum sequence length for generated text.
        """
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Generate sequences using the generator.
        Args:
            x (torch.Tensor): Input noise or seed tensor.
        Returns:
            torch.Tensor: Generated sequence logits.
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

    def generate(self, start_token, vocab_size, device="cpu"):
        """
        Generate synthetic text sequences.
        Args:
            start_token (int): Token to start the sequence.
            vocab_size (int): Size of the vocabulary.
            device (str): Device to run the model on ("cpu" or "cuda").
        Returns:
            list: Generated token sequence.
        """
        self.eval()
        with torch.no_grad():
            input_seq = torch.tensor([[start_token]], dtype=torch.long).to(device)
            generated_seq = [start_token]

            for _ in range(self.max_seq_len - 1):
                embedded = self.embedding(input_seq)
                lstm_out, _ = self.lstm(embedded)
                logits = self.fc(lstm_out[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).item()
                generated_seq.append(next_token)

                # Append the predicted token to the input sequence
                input_seq = torch.cat([input_seq, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

        return generated_seq


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Hidden dimension of LSTM layers.
        """
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification (real/fake)

    def forward(self, x):
        """
        Classify whether the input sequence is real or fake.
        Args:
            x (torch.Tensor): Input sequence.
        Returns:
            torch.Tensor: Probability of being real.
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out[:, -1, :])  # Use the last hidden state
        return torch.sigmoid(logits)


# TextGAN Class
class TextGAN:
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256, max_seq_len=50, device="cpu"):
        """
        Initialize the GAN components.
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension for LSTM.
            max_seq_len (int): Maximum sequence length for text generation.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        self.device = device
        self.vocab_size = vocab_size
        self.generator = Generator(vocab_size, embedding_dim, hidden_dim, max_seq_len).to(device)
        self.discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim).to(device)
        self.criterion = nn.BCELoss()
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, real_texts, num_epochs=100, batch_size=32):
        """
        Train the GAN.
        Args:
            real_texts (list): List of tokenized real texts.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        for epoch in range(num_epochs):
            for i in range(0, len(real_texts), batch_size):
                # Prepare real data
                real_batch = torch.tensor(real_texts[i:i + batch_size], dtype=torch.long).to(self.device)
                real_labels = torch.ones((real_batch.size(0), 1)).to(self.device)

                # Generate fake data
                noise = torch.randint(0, self.vocab_size, (real_batch.size(0), 1)).to(self.device)
                fake_batch = self.generator.generate(noise[:, 0].tolist(), self.vocab_size, device=self.device)
                fake_batch = torch.tensor(fake_batch, dtype=torch.long).to(self.device)
                fake_labels = torch.zeros((fake_batch.size(0), 1)).to(self.device)

                # Train Discriminator
                self.disc_optimizer.zero_grad()
                real_output = self.discriminator(real_batch)
                fake_output = self.discriminator(fake_batch)
                disc_loss = self.criterion(real_output, real_labels) + self.criterion(fake_output, fake_labels)
                disc_loss.backward()
                self.disc_optimizer.step()

                # Train Generator
                self.gen_optimizer.zero_grad()
                fake_output = self.discriminator(fake_batch)
                gen_loss = self.criterion(fake_output, real_labels)  # Trick the discriminator
                gen_loss.backward()
                self.gen_optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f}")

    def generate_synthetic_data(self, num_samples, start_token, vocab_size):
        """
        Generate synthetic hate speech data.
        Args:
            num_samples (int): Number of synthetic samples to generate.
            start_token (int): Token to start the sequence.
            vocab_size (int): Vocabulary size.
        Returns:
            list: List of synthetic token sequences.
        """
        synthetic_data = []
        for _ in range(num_samples):
            generated_seq = self.generator.generate(start_token, vocab_size, device=self.device)
            synthetic_data.append(generated_seq)
        return synthetic_data