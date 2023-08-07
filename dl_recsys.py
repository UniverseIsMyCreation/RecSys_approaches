from tools import *


def genre_count(movies):
    # create map between genre and idx
    genre2idx = dict()
    idx2genre = dict()
    idx = 0
    # search for each genre in dataset
    for line in movies['genres']:
        genres = list(map(lambda x: x.lower(), line.split('|')))
        for genre in genres:
            if genre2idx.get(genre, -1) == -1:
                genre2idx[genre] = idx
                idx2genre[idx] = genre
                idx += 1

    # save amount of genres
    genre_dim = idx
    
    # return result
    return genre2idx, idx2genre, genre_dim

class MovieLenDataset(Dataset):

    def __init__(self, dataset_rating, dataset_movie, mapping):
        super().__init__()
        
        # save our datasets
        self.rating = dataset_rating
        self.movie = dataset_movie
        
        self.genre2idx = mapping['genre2idx']
        self.idx2genre = mapping['idx2genre']
        self.genre_dim = mapping['genre_dim']

    def __len__(self):
        return self.rating.shape[0]
    
    def __getitem__(self, index):
        user_row = self.rating.iloc[index]
        
        genres = list(map(lambda x: x.lower(), self.movie.iloc[int(user_row['movieId'] - 1)]['genres'].split('|')))
        genre_vector = torch.zeros(self.genre_dim)
        for genre in genres:
            genre_vector[self.genre2idx[genre]] = 1
            
        return {
            'userId': torch.tensor(user_row['userId'] - 1, dtype=torch.int),
            'movieId': torch.tensor(user_row['movieId'] - 1, dtype=torch.int),
            'genre': genre_vector,
            'rating': torch.tensor(user_row['rating'], dtype=torch.float)
        }


class recsys_model(nn.Module):

    def __init__(self, max_user_amount, max_movie_amount, max_genre_amount, emb_dim=384, hid_dim=512):
        super().__init__()

        # emb layers for user and item
        self.user_ini_emb = nn.Embedding(max_user_amount, emb_dim)
        self.item_ini_emb = nn.Embedding(max_movie_amount, emb_dim)

        # hid layers for user and item
        self.user_fc = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim * 8),
            nn.ReLU(),
            nn.Linear(hid_dim * 8, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )

        self.item_fc = nn.Sequential(
            nn.Linear(emb_dim + max_genre_amount, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim * 8),
            nn.ReLU(),
            nn.Linear(hid_dim * 8, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim)
        )

    # forward pass
    def forward(self, users, movies, genres):
        users_emb, items_emb = self.user_ini_emb(users), self.item_ini_emb(movies)
        items_emb = torch.cat((items_emb, genres), dim=1)
        users_hid_emb, items_hid_emb = self.user_fc(users_emb), self.item_fc(items_emb)

        return users_hid_emb, items_hid_emb

# define loss
def recsys_loss(lambda1, lambda2, users_hid_emb, items_hid_emb, ratings):
    first_part = torch.mean((torch.sum(users_hid_emb * items_hid_emb, dim=1) - ratings) ** 2)
    second_part = lambda1 * torch.mean(torch.sum(users_hid_emb ** 2, dim=1))
    third_part = lambda2 * torch.mean(torch.sum(items_hid_emb ** 2, dim=1))
    
    return first_part + second_part + third_part


def get_loaders(ratings, movies, mapping, test_size=0.1, test_batch_size=128, train_batch_size=256):
    # split dataset on train and test
    from sklearn.model_selection import train_test_split
    train_ratings, test_ratings = train_test_split(ratings, test_size=test_size)

    # get torch dataset
    train_dataset = MovieLenDataset(train_ratings, movies, mapping)
    test_dataset = MovieLenDataset(test_ratings, movies, mapping)

    # get torch dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def train_model(model, train_loader, num_epochs, lr):
    # configure parameters and other stuff for learning
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    lambda1, lambda2 = 0.02, 0.02

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    loss_per_epoch = list()
    losses_per_epoch = dict()

    # set amount of epochs
    for epoch in range(num_epochs):
        
        loss_epoch = list()
        tepoch = tqdm(train_loader, unit='batch')
        # learn model through the torch dataloader
        for batch in train_loader:
            users = batch['userId'].to(device)
            movies = batch['movieId'].to(device)
            ratings = batch['rating'].to(device)
            genres = batch['genre'].to(device)

            # get models' outputs
            users_output, items_output = model(users, movies, genres)

            # calculate loss
            loss = recsys_loss(lambda1, lambda2, users_output, items_output, ratings)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update epoch information
            tepoch.update(1)
            tepoch.set_description(f'Epoch: {epoch + 1}')
            tepoch.set_postfix(recsys_loss=loss.item())

            # add loss information
            loss_epoch.append(loss.item())
            
        scheduler.step()
        
        losses_per_epoch[epoch + 1] = loss_epoch
        loss_per_epoch.append(np.mean(loss_epoch))
    
    return model, losses_per_epoch, loss_per_epoch


def evaluate_model(model, test_loader, show_preds=False):
    """
    evaluate recsys model
    show preds: if it's true, return dataframe with ground truth and what was predicted
    """
    from sklearn.metrics import mean_squared_error
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    rmse_results = list()
    preds = dict()
    preds['pred_rating'] = list()
    preds['rating'] = list()
    preds['userId'] = list()
    preds['movieId'] = list()
    show_preds_once = show_preds
    
    tepoch = tqdm(test_loader, unit='batch')
    for batch in test_loader:
        users = batch['userId'].to(device)
        movies = batch['movieId'].to(device)
        ratings = batch['rating'].numpy()
        genres = batch['genre'].to(device)

        # get models' outputs
        users_output, items_output = model(users, movies, genres)

        # calculate rating
        pred_ratings = torch.sum(users_output * items_output, dim=1).cpu().detach().numpy()
        rmse_metric = mean_squared_error(pred_ratings, ratings, squared=False)
        rmse_results.append(rmse_metric)
        
        if show_preds_once:
            preds['pred_rating'].extend(pred_ratings)
            preds['rating'].extend(ratings)
            preds['userId'].extend(users.cpu().detach().numpy())
            preds['movieId'].extend(movies.cpu().detach().numpy())
            show_preds_once = False

        # update epoch information
        tepoch.update(1)
    
    if show_preds:
        return np.mean(rmse_results), pd.DataFrame(preds).head(10) 

    return np.mean(rmse_results)


# get genres' information
genre2idx, idx2genre, genre_dim = genre_count(movies)
# movies - dataset with movies' info

# create mapping
mapping = dict()
mapping['genre2idx'] = genre2idx
mapping['idx2genre'] = idx2genre
mapping['genre_dim'] = genre_dim

# create loaders
train_loader, test_loader = get_loaders(ratings, movies, mapping)
# ratings - dataset with ratings
# movies - dataset with movies' info

# create recsys model
recsys = recsys_model(max_user_id, max_movie_id, genre_dim)
# max_user_id - eq amount of users
# max_movie_id - eq amount of movies

# train and get information about lossses
model, losses_per_epoch, loss_per_epoch = train_model(recsys, lr=0.001, num_epochs=2, train_loader=train_loader)

# check our model
rmse, pred_ratings = evaluate_model(recsys, test_loader, True)