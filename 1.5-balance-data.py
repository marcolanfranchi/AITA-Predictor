import pandas as pd 

def main():
    """
    This program balances the data by randomly sampling from the larger group to match the size of the smaller group.
    """
    df = pd.read_json('output/filtered_not_balanced.json.gz'  , compression='gzip')

    # Separate the groups
    asshole_group = df[df['link_flair_text'] == 'Asshole']
    not_asshole_group = df[df['link_flair_text'] == 'Not the A-hole']

    # Size of the smaller group
    asshole_size = len(asshole_group) # we know asshole group is smaller 

    # Randomly sample from the larger group to match the size of the smaller group
    sampled_not_asshole = not_asshole_group.sample(n=asshole_size)

    # Combine the samples with the smaller group
    balanced_sample = pd.concat([asshole_group, sampled_not_asshole])

    balanced_sample.to_json('output/filtered_and_balanced.json.gz' , compression='gzip')

if __name__ == '__main__':
    main()